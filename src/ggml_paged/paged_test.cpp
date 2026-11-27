// L5 test: exercises the GGML plugin's paged attention path (GGML_PAGED=YES) through GenAI's
// LLMPipeline and ContinuousBatchingPipeline, covering single-request agreement, multi-request
// batching isolation, copy-on-write via prefix caching and beam search (sync_forked_blocks),
// and optional ALL_LOGITS bucket regression.
//
// usage: paged_test <model.gguf> <d1.json> [GGML_BACKEND=...] [GGML_BUCKETS=8:d8_all.json]
//                   [MAX_TOKENS=24] [STOP_IDS=106,1]

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/runtime/core.hpp"

namespace {

constexpr size_t kDefaultBlockSize = 32;

struct TestContext {
    int passed = 0;
    int failed = 0;

    void check(const std::string& name, bool ok, const std::string& detail = "") {
        if (ok) {
            ++passed;
            std::cout << "[PASS] " << name << "\n";
        } else {
            ++failed;
            std::cout << "[FAIL] " << name;
            if (!detail.empty()) {
                std::cout << ": " << detail;
            }
            std::cout << "\n";
        }
    }

    void check_eq(const std::string& name, const std::string& expected, const std::string& actual) {
        if (expected == actual) {
            check(name, true);
        } else {
            check(name, false, "\n  expected: [" + expected + "]\n  actual  : [" + actual + "]");
        }
    }
};

std::string run_stateful(ov::CompiledModel& compiled_stateful,
                         const ov::genai::Tokenizer& tokenizer,
                         const std::string& prompt,
                         const ov::genai::GenerationConfig& cfg) {
    ov::genai::LLMPipeline pipe(compiled_stateful.create_infer_request(), tokenizer, cfg);
    auto res = pipe.generate(prompt, cfg);
    return res.texts.empty() ? std::string{} : res.texts.front();
}

void run_paged_suite(TestContext& ctx,
                     const std::string& label,
                     ov::CompiledModel& compiled_paged,
                     size_t expected_block_size,
                     const ov::genai::Tokenizer& tokenizer,
                     const ov::genai::GenerationConfig& greedy,
                     const std::vector<std::string>& batch_prompts,
                     const std::vector<std::string>& stateful_refs,
                     const std::string& pref_a,
                     const std::string& pref_b,
                     const std::string& ref_pref_b,
                     std::string* out_beam_top1 = nullptr) {
    // 0. Plugin properties advertised to GenAI's scheduler and KVCacheManager
    const auto max_blocks = compiled_paged.get_property("KV_CACHE_MAX_BLOCKS").as<size_t>();
    const auto block_size = compiled_paged.get_property("KV_CACHE_BLOCK_SIZE").as<size_t>();
    const auto kv_cap = compiled_paged.get_property("GGML_KV_CAPACITY").as<uint64_t>();
    std::cout << "\n=== " << label << " (KV_CACHE_MAX_BLOCKS=" << max_blocks
              << ", KV_CACHE_BLOCK_SIZE=" << block_size
              << ", GGML_KV_CAPACITY=" << kv_cap << ") ===\n";
    ctx.check(label + " / properties",
              max_blocks > 0 && block_size == expected_block_size &&
                  kv_cap == static_cast<uint64_t>(max_blocks) * block_size);

    // 1. Single-request LLMPipeline in paged mode (num_kv_blocks=0 relies on KV_CACHE_MAX_BLOCKS cap)
    {
        ov::genai::SchedulerConfig sched{};
        ov::genai::LLMPipeline pipe(compiled_paged, tokenizer, sched, greedy);
        auto res = pipe.generate(batch_prompts[0], greedy);
        const std::string actual = res.texts.empty() ? std::string{} : res.texts.front();
        ctx.check_eq(label + " / single paged vs stateful", stateful_refs[0], actual);
    }

    // 2. Multi-request continuous batching (3 prompts of different lengths in one batch)
    {
        ov::genai::SchedulerConfig sched{};
        ov::genai::ContinuousBatchingPipeline cb(compiled_paged, tokenizer, sched, greedy);
        std::vector<ov::genai::GenerationConfig> cfgs(batch_prompts.size(), greedy);
        auto results = cb.generate(batch_prompts, cfgs);
        ctx.check(label + " / batch result count", results.size() == batch_prompts.size());
        for (size_t i = 0; i < batch_prompts.size() && i < results.size(); ++i) {
            const std::string actual = results[i].m_generation_ids.empty() ? std::string{} : results[i].m_generation_ids[0];
            ctx.check_eq(label + " / batch[" + std::to_string(i) + "] vs stateful", stateful_refs[i], actual);
        }
    }

    // 3. Prefix caching + CoW: shared prefix > 32 tokens (1 full block + partial block) then diverge.
    //    If sync_forked_blocks / copy_cache_slots fails to copy the shared partial block, pref_b diverges.
    {
        ov::genai::SchedulerConfig sched{};
        sched.enable_prefix_caching = true;
        ov::genai::ContinuousBatchingPipeline cb(compiled_paged, tokenizer, sched, greedy);

        // Warm prefix cache with pref_a, then run pref_b sequentially, then both concurrently.
        cb.generate(std::vector<std::string>{pref_a}, std::vector<ov::genai::GenerationConfig>{greedy});
        auto res_seq =
            cb.generate(std::vector<std::string>{pref_b}, std::vector<ov::genai::GenerationConfig>{greedy});
        const std::string actual_seq = (!res_seq.empty() && !res_seq[0].m_generation_ids.empty())
                                           ? res_seq[0].m_generation_ids[0]
                                           : std::string{};
        ctx.check_eq(label + " / prefix-cache sequential vs stateful", ref_pref_b, actual_seq);

        auto res_conc = cb.generate(std::vector<std::string>{pref_a, pref_b},
                                    std::vector<ov::genai::GenerationConfig>{greedy, greedy});
        const std::string actual_conc = (res_conc.size() == 2 && !res_conc[1].m_generation_ids.empty())
                                            ? res_conc[1].m_generation_ids[0]
                                            : std::string{};
        ctx.check_eq(label + " / prefix-cache concurrent vs stateful", ref_pref_b, actual_conc);
    }

    // 4. Beam search CoW: num_beams=3 on a non-multiple-of-32 prompt forces partial-block forks.
    {
        ov::genai::SchedulerConfig sched{};
        ov::genai::ContinuousBatchingPipeline cb(compiled_paged, tokenizer, sched, greedy);
        ov::genai::GenerationConfig beam_cfg = greedy;
        beam_cfg.do_sample = false;
        beam_cfg.num_beams = 3;
        beam_cfg.num_return_sequences = 3;

        auto results = cb.generate(std::vector<std::string>{batch_prompts[1]},
                                   std::vector<ov::genai::GenerationConfig>{beam_cfg});
        const bool have_three = results.size() == 1 && results[0].m_generation_ids.size() == 3;
        bool all_non_empty = have_three;
        if (have_three) {
            for (size_t i = 0; i < 3; ++i) {
                if (results[0].m_generation_ids[i].empty()) {
                    all_non_empty = false;
                }
                std::cout << "  beam[" << i << "]: " << results[0].m_generation_ids[i] << "\n";
            }
            if (out_beam_top1) {
                *out_beam_top1 = results[0].m_generation_ids[0];
            }
        }
        ctx.check(label + " / beam search (3 beams non-empty)", all_non_empty);
    }
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <model.gguf> <d1.json> [GGML_BACKEND=...] [GGML_BUCKETS=8:d8_all.json]"
                  << " [MAX_TOKENS=24] [STOP_IDS=106,1]\n";
        return 1;
    }

    const std::string gguf = argv[1];
    const std::string d1 = argv[2];

    ov::genai::GenerationConfig greedy;
    greedy.max_new_tokens = 24;
    greedy.do_sample = false;

    ov::AnyMap base_props{{"GGML_WEIGHTS", gguf}};
    std::string buckets;

    for (int i = 3; i < argc; ++i) {
        const std::string kv = argv[i];
        const auto eq = kv.find('=');
        if (eq == std::string::npos) {
            throw std::invalid_argument("expected KEY=value, got " + kv);
        }
        const std::string k = kv.substr(0, eq);
        const std::string v = kv.substr(eq + 1);
        if (k == "MAX_TOKENS") {
            greedy.max_new_tokens = std::stoul(v);
        } else if (k == "STOP_IDS") {
            std::stringstream ids(v);
            for (std::string id; std::getline(ids, id, ',');) {
                greedy.stop_token_ids.insert(std::stoll(id));
            }
        } else if (k == "GGML_BUCKETS") {
            buckets = v;
        } else if (k == "GGML_PAGED") {
            // Controlled by the test itself.
        } else {
            base_props[k] = v;
        }
    }

    const std::vector<std::string> batch_prompts = {
        "The capital of France is",
        "List three primary colors in painting and explain briefly why they are called primary:",
        "In computer science, a hash table is a data structure that",
    };

    // > 40 tokens so the shared prefix spans at least one full 32-token KV block plus a partial block.
    const std::string shared_prefix =
        "You are a concise geography assistant. Answer each question in one short sentence without extra "
        "commentary. Keep every answer factual, direct, and easy to verify. Question: What is the capital city of ";
    const std::string pref_a = shared_prefix + "Japan?";
    const std::string pref_b = shared_prefix + "Italy?";

    ov::Core core;
    ov::genai::Tokenizer tokenizer(gguf);

    // Build stateful baseline once (shares weights with paged compiles via L2 registry).
    auto compiled_stateful = core.compile_model(d1, "GGML", base_props);
    std::vector<std::string> stateful_refs;
    stateful_refs.reserve(batch_prompts.size());
    for (const auto& p : batch_prompts) {
        stateful_refs.push_back(run_stateful(compiled_stateful, tokenizer, p, greedy));
    }
    const std::string ref_pref_b = run_stateful(compiled_stateful, tokenizer, pref_b, greedy);

    TestContext ctx;

    // Paged d1-only pass (default block size 32)
    ov::AnyMap paged_props = base_props;
    paged_props["GGML_PAGED"] = "YES";
    auto compiled_paged = core.compile_model(d1, "GGML", paged_props);
    std::string beam_d1_top1;
    run_paged_suite(ctx,
                    "paged d1 (bs=32)",
                    compiled_paged,
                    kDefaultBlockSize,
                    tokenizer,
                    greedy,
                    batch_prompts,
                    stateful_refs,
                    pref_a,
                    pref_b,
                    ref_pref_b,
                    &beam_d1_top1);

    // Fail-fast check: cache eviction is incompatible with device-managed fixed-size KV cache
    {
        ov::genai::SchedulerConfig sched{};
        sched.use_cache_eviction = true;
        bool threw = false;
        try {
            ov::genai::ContinuousBatchingPipeline cb(compiled_paged, tokenizer, sched, greedy);
        } catch (const std::exception& e) {
            threw = true;
            std::cout << "  eviction fail-fast error: " << e.what() << "\n";
        }
        ctx.check("paged d1 / fail-fast on use_cache_eviction", threw);
    }

    // Negotiated non-default block size (16) proves GenAI reads KV_CACHE_BLOCK_SIZE instead of assuming 32
    {
        ov::AnyMap paged_bs16_props = paged_props;
        paged_bs16_props["GGML_KV_BLOCK_SIZE"] = "16";
        auto compiled_bs16 = core.compile_model(d1, "GGML", paged_bs16_props);
        run_paged_suite(ctx,
                        "paged d1 (bs=16)",
                        compiled_bs16,
                        16,
                        tokenizer,
                        greedy,
                        batch_prompts,
                        stateful_refs,
                        pref_a,
                        pref_b,
                        ref_pref_b);
    }

    // Optional paged + ALL_LOGITS bucket pass
    if (!buckets.empty()) {
        ov::AnyMap bucket_props = paged_props;
        bucket_props["GGML_BUCKETS"] = buckets;
        auto compiled_bucket = core.compile_model(d1, "GGML", bucket_props);
        std::string beam_bucket_top1;
        run_paged_suite(ctx,
                        "paged d1+buckets (bs=32)",
                        compiled_bucket,
                        kDefaultBlockSize,
                        tokenizer,
                        greedy,
                        batch_prompts,
                        stateful_refs,
                        pref_a,
                        pref_b,
                        ref_pref_b,
                        &beam_bucket_top1);
        ctx.check_eq("paged d1+buckets / beam top1 vs paged d1", beam_d1_top1, beam_bucket_top1);
    }

    std::cout << "\nSummary: " << ctx.passed << " passed, " << ctx.failed << " failed\n";
    return ctx.failed == 0 ? 0 : 2;
} catch (const std::exception& e) {
    std::cerr << "\nerror: " << e.what() << "\n";
    return 1;
}
