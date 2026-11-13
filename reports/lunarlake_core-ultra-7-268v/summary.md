# GGML device vs OpenVINO: summary

```
OS: Windows-11-10.0.26200-SP0  Date: 2026-11-12 17:07
CPU: Intel(R) Core(TM) Ultra 7 268V x1 (8 cores, 8 threads)
RAM: 32 GB
GPU: Microsoft Remote Display Adapter, driver 10.0.26100.9818
GPU: Intel(R) Arc(TM) 140V GPU (16GB), driver 32.0.101.8425
vulkaninfo --summary:
  Vulkan Instance Version: 1.4.350
  VK_LAYER_LUNARG_crash_diagnostic  Crash Diagnostic Layer is a crash/hang debugging tool that helps determines GPU progress in a Vulkan application. 1.4.350  version 1
  GPU0:
  apiVersion         = 1.4.333
  deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
  deviceName         = Intel(R) Arc(TM) 140V GPU (16GB)
  driverName         = Intel Corporation
  driverInfo         = 101.8425
  GPU1:
  apiVersion         = 1.4.333
  deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
  deviceName         = Intel(R) Arc(TM) 140V GPU (16GB)
  driverName         = Intel Corporation
  driverInfo         = 101.8425
OpenVINO devices: CPU (Intel(R) Core(TM) Ultra 7 268V), GGML (ggml), GPU (Intel(R) Arc(TM) 140V GPU (16GB) (iGPU))
OpenVINO GPU baseline: GPU
```

GGML = OpenVINO's GGML device replaying llama.cpp's graph (CPU or Vulkan backend). OV = GenAI's own GGUF path on an OpenVINO plugin. All runs are warm, greedy, 32 new tokens.

## Decode: time per token (TPOT, ms)

| Model | GiB | GGML CPU | GGML GPU | OV CPU | OV GPU | GGML GPU vs OV CPU | GGML GPU vs OV GPU | Notes |
| :-- | --: | --: | --: | --: | --: | --: | --: | :-- |
| granite-4.0-350m-Q4_K_M | 0.22 | 6.2 | 8.2 |  |  |  |  | GenAI GGUF: architecture not supported |
| Qwen_Qwen3-0.6B-Q4_K_M | 0.45 | 9.5 | 10.9 | 13.3 | 9.3 | 1.2x | 0.9x |  |
| Qwen_Qwen3.5-0.8B-Q4_K_M | 0.54 | 14.8 | 14.8 |  |  |  |  | GenAI GGUF: tensor type not supported; no prefill buckets for recurrent models (by design) |
| Llama-3.2-1B-Instruct-Q4_K_M | 0.75 | 15.6 | 13.6 | 26.2 | 14.7 | 1.9x | 1.1x |  |
| granite-4.0-1b-Q4_K_M | 0.95 | 20.8 | 19.6 |  |  |  |  | GenAI GGUF: architecture not supported |
| smollm2-1.7b-instruct-q4_k_m | 0.98 | 20.5 | 16.2 | 33.6 | 19.3 | 2.1x | 1.2x |  |
| qwen2.5-1.5b-instruct-q4_k_m | 1.04 | 19.4 | 17 | 31.7 | 18.7 | 1.9x | 1.1x |  |
| Qwen_Qwen3-1.7B-Q4_K_M | 1.19 | 21.4 | 18.3 | 33.9 | 17.8 | 1.9x | 1.0x |  |
| Qwen_Qwen3.5-2B-Q4_K_M | 1.30 | 28.1 | 23.5 |  |  |  |  | GenAI GGUF: tensor type not supported; no prefill buckets for recurrent models (by design) |
| Ministral-3b-instruct.Q4_K_M | 1.86 | 35.3 | 22.6 | 57.5 | 25 | 2.5x | 1.1x |  |
| Llama-3.2-3B-Instruct-Q4_K_M | 1.88 | 37.3 | 30.5 | 60.2 | 35 | 2.0x | 1.1x |  |
| granite-4.0-micro-Q4_K_M | 1.96 | 40.7 | 31.3 |  |  |  |  | GenAI GGUF: architecture not supported |
| Phi-3-mini-4k-instruct-Q4_K_M | 2.23 | 44.6 | 28.6 |  |  |  |  | GenAI GGUF: tensor type not supported |
| Phi-3.5-mini-instruct-Q4_K_M | 2.23 | 44.7 | 28.7 |  |  |  |  | GenAI GGUF: tensor type not supported |
| gemma-3-4b-it-Q4_K_M | 2.32 | 45.9 | 35 |  |  |  |  | GenAI GGUF: architecture not supported |
| microsoft_Phi-4-mini-instruct-Q4_K_M | 2.32 | 45 | 33.8 |  |  |  |  | GenAI GGUF: tensor type not supported |
| Qwen3-4B-Q4_K_M | 2.33 | 46.9 | 34.4 | 71.7 | 36.8 | 2.1x | 1.1x |  |
| Qwen_Qwen3.5-4B-Q4_K_M | 2.81 | 59 | 42.3 |  |  |  |  | GenAI GGUF: tensor type not supported; no prefill buckets for recurrent models (by design) |
| google_gemma-4-E2B-it-Q4_K_M | 3.22 | 33.3 |  |  |  |  |  | GenAI GGUF: tensor type not supported; ggml_vulkan: 0 = Intel(R) Arc(TM) 140V GPU (16GB) (Intel Corporation) | uma: 1 | |
| Mistral-7B-Instruct-v0.3-Q4_K_M | 4.07 | 77.4 | 49.2 | 131.1 | 63.6 | 2.7x | 1.3x |  |
| tencent_Hunyuan-7B-Instruct-Q4_K_M | 4.30 | 83.6 | 53.2 |  |  |  |  | GenAI GGUF: architecture not supported |
| ggml-model-Q4_K_M | 4.36 | 76.7 | 49.7 | 144.4 | 63.9 | 2.9x | 1.3x |  |
| DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 4.36 | 76.2 | 50.3 | 124.8 | 64.2 | 2.5x | 1.3x |  |
| qwen2.5-coder-7b-instruct-q4_k_m | 4.36 | 78 | 51.1 | 125.4 | 64.8 | 2.5x | 1.3x |  |
| EXAONE-3.5-7.8B-Instruct-Q4_K_M | 4.44 | 80.2 | 51.8 |  |  |  |  | GenAI GGUF: architecture not supported |
| Ministral-8B-Instruct-2410-Q4_K_M | 4.57 | 82.9 | 53.4 | 138.4 | 73.9 | 2.6x | 1.4x |  |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | 4.58 | 82 | 53.2 | 138.7 | 71 | 2.6x | 1.3x |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 4.58 | 82 | 56.6 | 143.5 | 69.3 | 2.5x | 1.2x |  |
| granite-3.2-8b-instruct-Q4_K_M | 4.60 | 89.5 | 58 |  |  |  |  | GenAI GGUF: architecture not supported |
| Qwen3-8B-Q4_K_M | 4.68 | 84 | 54.7 | 139.6 | 73.4 | 2.6x | 1.3x |  |
| prism-ml_Bonsai-8B-unpacked-Q4_K_M | 4.84 | 88.4 | 58.5 |  |  |  |  | GenAI GGUF: tensor type not supported |
| google_gemma-4-E4B-it-Q4_K_M | 5.03 | 58.8 |  |  |  |  |  | GenAI GGUF: tensor type not supported; ggml_vulkan: 0 = Intel(R) Arc(TM) 140V GPU (16GB) (Intel Corporation) | uma: 1 | |
| Qwen3.5-9B-Q4_K_M | 5.24 | 96.4 | 63.2 |  |  |  |  | GenAI GGUF: tensor type not supported; no prefill buckets for recurrent models (by design) |
| gemma-4-12B-it-Q4_K_M | 7.14 | 139.9 | 89.9 |  |  |  |  | GenAI GGUF: tensor type not supported |

Ratios above 1 mean the GGML GPU path decodes faster.

## First token (TTFT, ms)

d1 prefills one token per step; b8 uses the 8-token bucket.

| Model | Prompt tokens | GGML CPU d1 | GGML CPU b8 | GGML GPU d1 | GGML GPU b8 | OV CPU | OV GPU |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| granite-4.0-350m-Q4_K_M | 35 | 192.4 | 92.5 | 226.3 | 47.7 |  |  |
| Qwen_Qwen3-0.6B-Q4_K_M | 13 | 110.5 | 60.9 | 113.7 | 29.4 | 34.7 | 26.1 |
| Qwen_Qwen3.5-0.8B-Q4_K_M | 17 | 249.7 |  | 199.2 |  |  |  |
| Llama-3.2-1B-Instruct-Q4_K_M | 40 | 604.1 | 270.3 | 470.4 | 92.8 | 47.9 | 113.5 |
| granite-4.0-1b-Q4_K_M | 35 | 712.8 | 400.9 | 605.3 | 146.1 |  |  |
| smollm2-1.7b-instruct-q4_k_m | 35 | 705.5 | 422.5 | 541.2 | 135.3 | 52 | 104.9 |
| qwen2.5-1.5b-instruct-q4_k_m | 34 | 626.2 | 366.5 | 508 | 132.9 | 43 | 100.9 |
| Qwen_Qwen3-1.7B-Q4_K_M | 13 | 267.5 | 156.4 | 222.6 | 61.6 | 93.8 | 28.3 |
| Qwen_Qwen3.5-2B-Q4_K_M | 17 | 476.6 |  | 338.1 |  |  |  |
| Ministral-3b-instruct.Q4_K_M | 14 | 481.9 | 287.5 | 304.6 | 88 | 153.4 | 32.2 |
| Llama-3.2-3B-Instruct-Q4_K_M | 40 | 1473.7 | 736.5 | 1098.3 | 237.5 | 117.5 | 172.5 |
| granite-4.0-micro-Q4_K_M | 35 | 1379 | 828 | 1011.4 | 263.9 |  |  |
| Phi-3-mini-4k-instruct-Q4_K_M | 15 | 645.6 | 422.4 | 420.3 | 110.4 |  |  |
| Phi-3.5-mini-instruct-Q4_K_M | 14 | 624 | 413.2 | 394.7 | 112.1 |  |  |
| gemma-3-4b-it-Q4_K_M | 14 | 633.7 | 344.3 | 456.9 | 115.9 |  |  |
| microsoft_Phi-4-mini-instruct-Q4_K_M | 8 | 351 | 177.9 | 246.7 | 59.7 |  |  |
| Qwen3-4B-Q4_K_M | 13 | 592.1 | 376.9 | 420.7 | 121 | 191.5 | 49.7 |
| Qwen_Qwen3.5-4B-Q4_K_M | 15 | 891.5 |  | 577.7 |  |  |  |
| google_gemma-4-E2B-it-Q4_K_M | 14 | 444.7 | 117.7 |  |  |  |  |
| Mistral-7B-Instruct-v0.3-Q4_K_M | 10 | 798.7 | 686.8 | 490.3 | 200.7 | 287.7 | 79.4 |
| tencent_Hunyuan-7B-Instruct-Q4_K_M | 7 | 574.9 | 339.7 | 365.3 | 106.2 |  |  |
| ggml-model-Q4_K_M | 24 | 1854 | 975.5 | 1140.1 | 312.4 | 508.4 | 176.7 |
| DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 8 | 606 | 317.2 | 379.9 | 107.1 | 238.4 | 66.8 |
| qwen2.5-coder-7b-instruct-q4_k_m | 34 | 2633.5 | 1705.1 | 1662.2 | 525.2 | 153 | 198.2 |
| EXAONE-3.5-7.8B-Instruct-Q4_K_M | 24 | 1915.1 | 1033.9 | 1202.2 | 305.5 |  |  |
| Ministral-8B-Instruct-2410-Q4_K_M | 8 | 659.8 | 339.8 | 408.5 | 108.9 | 249.6 | 78.4 |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | 8 | 645 | 335.5 | 405.9 | 105.2 | 266.3 | 74.3 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 40 | 3244.4 | 1828 | 2125.1 | 523.6 | 283.6 | 205 |
| granite-3.2-8b-instruct-Q4_K_M | 65 | 5747.3 | 3417.4 | 3612.9 | 979.2 |  |  |
| Qwen3-8B-Q4_K_M | 13 | 1089.9 | 703.6 | 674.4 | 213.2 | 341.2 | 91.2 |
| prism-ml_Bonsai-8B-unpacked-Q4_K_M | 17 | 1469.4 | 828.6 | 941.3 | 273.9 |  |  |
| google_gemma-4-E4B-it-Q4_K_M | 14 | 809.1 | 311.6 |  |  |  |  |
| Qwen3.5-9B-Q4_K_M | 15 | 1442.3 |  | 890.4 |  |  |  |
| gemma-4-12B-it-Q4_K_M | 18 | 2482.8 | 1822.3 | 1564.8 | 528.1 |  |  |

## Same output text

GGML GPU vs GGML CPU compares one graph on two backends; late differences are float rounding. OV CPU uses a u8 KV cache and int8 activations by default, so it often drifts from GGML after a few tokens.

| Model | GGML GPU = GGML CPU | GGML CPU = OV CPU | GGML GPU = OV GPU |
| :-- | :-: | :-: | :-: |
| granite-4.0-350m-Q4_K_M | no |  |  |
| Qwen_Qwen3-0.6B-Q4_K_M | yes | no | no |
| Qwen_Qwen3.5-0.8B-Q4_K_M | no |  |  |
| Llama-3.2-1B-Instruct-Q4_K_M | yes | yes | yes |
| granite-4.0-1b-Q4_K_M | yes |  |  |
| smollm2-1.7b-instruct-q4_k_m | yes | yes | yes |
| qwen2.5-1.5b-instruct-q4_k_m | yes | no | no |
| Qwen_Qwen3-1.7B-Q4_K_M | yes | no | no |
| Qwen_Qwen3.5-2B-Q4_K_M | yes |  |  |
| Ministral-3b-instruct.Q4_K_M | yes | no | no |
| Llama-3.2-3B-Instruct-Q4_K_M | yes | no | no |
| granite-4.0-micro-Q4_K_M | no |  |  |
| Phi-3-mini-4k-instruct-Q4_K_M | no |  |  |
| Phi-3.5-mini-instruct-Q4_K_M | no |  |  |
| gemma-3-4b-it-Q4_K_M | yes |  |  |
| microsoft_Phi-4-mini-instruct-Q4_K_M | no |  |  |
| Qwen3-4B-Q4_K_M | no | yes | no |
| Qwen_Qwen3.5-4B-Q4_K_M | yes |  |  |
| google_gemma-4-E2B-it-Q4_K_M |  |  |  |
| Mistral-7B-Instruct-v0.3-Q4_K_M | yes | no | no |
| tencent_Hunyuan-7B-Instruct-Q4_K_M | yes |  |  |
| ggml-model-Q4_K_M | yes | no | no |
| DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | yes | yes | yes |
| qwen2.5-coder-7b-instruct-q4_k_m | yes | yes | yes |
| EXAONE-3.5-7.8B-Instruct-Q4_K_M | yes |  |  |
| Ministral-8B-Instruct-2410-Q4_K_M | yes | yes | no |
| DeepSeek-R1-Distill-Llama-8B-Q4_K_M | yes | yes | no |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | yes | yes | yes |
| granite-3.2-8b-instruct-Q4_K_M | yes |  |  |
| Qwen3-8B-Q4_K_M | yes | no | no |
| prism-ml_Bonsai-8B-unpacked-Q4_K_M | no |  |  |
| google_gemma-4-E4B-it-Q4_K_M |  |  |  |
| Qwen3.5-9B-Q4_K_M | yes |  |  |
| gemma-4-12B-it-Q4_K_M | yes |  |  |

Per-run details (text, load time, token counts): cpu/e2e.md and gpu/e2e.md; each run's full output is in the .log files next to them.
