// Runs a dumped llama.cpp decoder graph on the OpenVINO "GGML" device through GenAI's LLMPipeline.
// Passing a device (CPU, GPU) instead of an artifact runs the same .gguf through GenAI's own GGUF path.
//
// usage: ggml_llm <model.gguf> <artifact.json|device> "<prompt>" [max_new_tokens] [GGML_KEY=value ...]
//   e.g. GGML_BACKEND=Vulkan0, GGML_BUCKETS=8:decoder_b8.json, GGML_PAGED=YES (continuous batching);
//   STOP_IDS=106,1 sets GenAI stop tokens instead; WARMUP=1 runs an untimed generate first.

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/runtime/core.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " <model.gguf> <artifact.json|device> \"<prompt>\" [max_new_tokens] [GGML_KEY=value ...]\n";
        return 1;
    }
    const std::string gguf = argv[1], target = argv[2], prompt = argv[3];
    ov::genai::GenerationConfig config;
    config.max_new_tokens = argc > 4 ? std::stoul(argv[4]) : 32;
    ov::AnyMap properties{{"GGML_WEIGHTS", gguf}};
    bool warmup = false;
    for (int i = 5; i < argc; ++i) {
        const std::string kv = argv[i];
        const auto eq = kv.find('=');
        if (eq == std::string::npos) {
            throw std::invalid_argument("expected KEY=value, got " + kv);
        }
        if (kv.substr(0, eq) == "STOP_IDS") {  // generation config, not a plugin property: "106,1"
            std::stringstream ids(kv.substr(eq + 1));
            for (std::string id; std::getline(ids, id, ',');) {
                config.stop_token_ids.insert(std::stoll(id));
            }
            continue;
        }
        if (kv.substr(0, eq) == "WARMUP") {  // untimed generate first, e.g. for lazy GPU pipeline creation
            warmup = kv.substr(eq + 1) != "0";
            continue;
        }
        properties[kv.substr(0, eq)] = kv.substr(eq + 1);
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<ov::genai::LLMPipeline> pipe;
    const bool artifact = target.size() > 5 && target.compare(target.size() - 5, 5, ".json") == 0;
    if (!artifact) {  // an OpenVINO device: GenAI's own GGUF path
        pipe = std::make_unique<ov::genai::LLMPipeline>(gguf, target);
    } else {
        ov::Core core;
        auto compiled = core.compile_model(target, "GGML", properties);
        ov::genai::Tokenizer tokenizer(gguf);
        if (properties.count("GGML_PAGED")) {  // the paged runner reports its block capacity itself
            pipe = std::make_unique<ov::genai::LLMPipeline>(compiled, tokenizer, ov::genai::SchedulerConfig{}, config);
        } else {
            pipe = std::make_unique<ov::genai::LLMPipeline>(compiled.create_infer_request(), tokenizer, config);
        }
    }
    const auto load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (warmup) {
        ov::genai::GenerationConfig short_config = config;
        short_config.max_new_tokens = 2;
        pipe->generate(prompt, short_config);
    }
    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };
    auto result = pipe->generate(prompt, config, streamer);
    auto& m = result.perf_metrics;
    std::cout << "\n\nload " << load_ms << " ms, prompt " << m.get_num_input_tokens() << " tokens, generated "
              << m.get_num_generated_tokens() << " tokens, TTFT " << m.get_ttft().mean << " ms, TPOT "
              << m.get_tpot().mean << " ms, " << m.get_throughput().mean << " tokens/s" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "\nerror: " << e.what() << "\n";
    return 1;
}
