// L4 test: two InferRequests of one GGML compiled model must keep independent KV state.
//
// Drives the plugin's stateful decoder directly (no GenAI): prefill + one decode on request A,
// the same on request B with different tokens, then a second decode on request A. With one
// shared runner (pre-fix), B's prefill overwrites A's KV slots, so A's second-decode logits
// diverge from a single-request baseline that never ran B. Post-fix they must be identical.
//
// usage: requests_test <model.gguf> <artifact.json> [GGML_BACKEND=...]

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"

namespace {
// One prefill (tokens) or decode (one token) step on the request; returns the logits row.
std::vector<float> step(ov::InferRequest& req, const std::vector<int64_t>& tokens, size_t seq_len) {
    req.set_tensor("input_ids", ov::Tensor(ov::element::i64, {1, tokens.size()}, tokens.data()));
    std::vector<int64_t> mask(seq_len, 1);
    req.set_tensor("attention_mask", ov::Tensor(ov::element::i64, {1, seq_len}, mask.data()));
    std::vector<int32_t> beam{0};
    req.set_tensor("beam_idx", ov::Tensor(ov::element::i32, {1}, beam.data()));
    req.infer();
    const auto out = req.get_output_tensor();
    const size_t vocab = out.get_shape()[2];
    const float* logits = static_cast<const float*>(out.data());
    // one row: the last (only) position
    return std::vector<float>(logits, logits + vocab);
}

size_t argmax(const std::vector<float>& v) {
    return static_cast<size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); i++) {
        d = std::max(d, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return d;
}
}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <artifact.json> [GGML_BACKEND=...] [CONSTRUCT_ONLY=1]\n";
        return 1;
    }
    ov::Core core;
    ov::AnyMap properties{{"GGML_WEIGHTS", argv[1]}};
    bool construct_only = false;
    std::string prompt_ids_file, logits_out;
    int gen = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string kv = argv[i];
        const std::string key = kv.substr(0, kv.find('='));
        const std::string value = kv.substr(kv.find('=') + 1);
        if (key == "CONSTRUCT_ONLY") {
            construct_only = value != "0";
            continue;
        }
        if (key == "PROMPT_IDS") {
            prompt_ids_file = value;
            continue;
        }
        if (key == "LOGITS_OUT") {
            logits_out = value;
            continue;
        }
        if (key == "GEN") {
            gen = std::stoi(value);
            continue;
        }
        properties[key] = value;
    }
    auto compiled = core.compile_model(argv[2], "GGML", properties);

    // Long-sequence mode (L8): prefill the whole token list in one infer() -- the runner
    // chunks internally and wraps KV slots once the sequence passes the capacity -- and dump
    // the logits row of the last position for comparison against llama.cpp.
    if (!prompt_ids_file.empty()) {
        std::ifstream f(prompt_ids_file);
        std::vector<int64_t> ids;
        for (int64_t id; f >> id;) {
            ids.push_back(id);
        }
        if (ids.empty()) {
            std::cerr << "no ids in " << prompt_ids_file << "\n";
            return 1;
        }
        std::cout << "GGML_KV_CAPACITY " << compiled.get_property("GGML_KV_CAPACITY").as<uint64_t>() << std::endl;
        auto req = compiled.create_infer_request();
        auto logits = step(req, ids, ids.size());
        const size_t arg = argmax(logits);
        std::cout << "prompt " << ids.size() << " tokens, argmax " << arg << std::endl;
        if (!logits_out.empty()) {
            std::ofstream o(logits_out, std::ios::binary);
            o.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
        }
        // Greedy continuation: decode GEN tokens from the same (ringed) state, printing each id.
        size_t next = argmax(logits);
        for (int g = 0; g < gen; ++g) {
            logits = step(req, {static_cast<int64_t>(next)}, ids.size() + g + 1);
            next = argmax(logits);
            std::cout << next << " ";
        }
        if (gen) {
            std::cout << "\n";
        }
        return 0;
    }

    if (construct_only) {
        // Exercise the per-request build path (whisper decoders build a fresh cross graph per
        // request); no tensors are fed.
        auto reqA = compiled.create_infer_request();
        auto reqB = compiled.create_infer_request();
        std::cout << "two requests constructed" << std::endl;
        return 0;
    }

    const std::vector<int64_t> tokens_a = {101, 2023, 2003, 2190, 1011, 1997, 2605};  // whatever ids: values feed embeddings only
    const std::vector<int64_t> tokens_b = {400, 500, 600, 700, 800, 900, 1000, 1100};
    const std::vector<int64_t> next_a = {333}, next_b = {444};

    // Baseline: one request, never disturbed.
    auto req0 = compiled.create_infer_request();
    auto logits_prefill_a = step(req0, tokens_a, tokens_a.size());
    const auto t_a1 = argmax(logits_prefill_a);
    auto logits_decode_a1 = step(req0, next_a, tokens_a.size() + 1);
    auto logits_decode_a2 = step(req0, next_a, tokens_a.size() + 2);

    // Two-request scenario: A, B, then A again.
    auto reqA = compiled.create_infer_request();
    auto reqB = compiled.create_infer_request();
    step(reqA, tokens_a, tokens_a.size());
    step(reqB, tokens_b, tokens_b.size());
    step(reqB, next_b, tokens_b.size() + 1);
    step(reqA, next_a, tokens_a.size() + 1);
    auto logits_decode_a2_two = step(reqA, next_a, tokens_a.size() + 2);

    const double d = max_diff(logits_decode_a2, logits_decode_a2_two);
    const bool same_argmax = argmax(logits_decode_a2) == argmax(logits_decode_a2_two);
    std::cout << "baseline A decode2 argmax " << argmax(logits_decode_a2) << " vs two-request "
              << argmax(logits_decode_a2_two) << ", max |diff| " << d
              << (d == 0.0 && same_argmax ? " -> ISOLATED" : " -> CONTAMINATED") << std::endl;
    return d == 0.0 && same_argmax ? 0 : 2;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
