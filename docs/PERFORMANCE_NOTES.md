# Performance notes: Four GGUF paths on Intel integrated graphics

Latest recorded batch for each model and quantization as of 2026-11-19. Batch IDs identify the source summaries in `llama-work-note/results/compare-gpu-4paths`.

## Shared configuration

- Windows, Intel Core Ultra 9 285K. All paths target its Intel integrated GPU: `Vulkan1` for Vulkan and `GPU.0` for OpenVINO.
- Context: 4096 tokens. llama.cpp microbatch: `-ub 1024`. GGML graph artifacts: `n_kv=4096`, with a one-token decode graph and a 1024-token prefill bucket.
- Each run requests 32 output tokens, uses greedy decoding, ignores EOS, and does not apply a chat template. Each case has one warm-up; measured repetitions are listed per model.
- Both llama.cpp paths and the GGML graph path use flash attention and f16 KV. GenAI + OV uses OpenVINO GPU plugin defaults.

| Path | Runtime and model input |
| --- | --- |
| llama.cpp + Vulkan | llama.cpp with the Vulkan backend, reading GGUF |
| llama.cpp + OV | llama.cpp with the OpenVINO backend, reading GGUF |
| GenAI + Vulkan | OpenVINO GenAI with the GGML device on Vulkan, replaying exported GGML graphs |
| GenAI + OV | OpenVINO GenAI on the OpenVINO GPU plugin; the GGUF reader or frontend is specified per model |

Values are medians of measured runs. Ratios in parentheses use llama.cpp + Vulkan as `1.00x` for the same model, prompt, and metric. **TTFT (ms)** covers prompt inference up to the first token's logits; **TPOT (ms per output token)** covers subsequent inference per token. The timings use llama.cpp's prompt/eval times and GenAI's raw inference durations, excluding model load, sampling, and pipeline overhead. Prompt labels are nominal. The separate prompt token tables report actual counts from each runtime log; `-` means no count was recorded.

## Llama-3.2-1B-Instruct

Both batches used llama.cpp `20260904-64a155`, OpenVINO `20261111-e65c59`, GenAI `20261026-e2a691`, and graph generator `20261117-03349b`. GenAI + OV used GenAI's built-in GGUF reader. Both batches used five measured runs per case and completed all four paths at every prompt length.

### Q4_0 (requantized)

Batch: `20261117-155418`.

**Prompt tokens**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 6 | 6 | 5 | 5 |
| 250 | 238 | 238 | 237 | 237 |
| 1000 | 971 | 971 | 970 | 970 |
| 2000 | 1937 | 1937 | 1936 | 1936 |

**TTFT (ms)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 29.2 (1.00x) | 36.2 (1.24x) | 1240.0 (42.39x) | 27.6 (0.94x) |
| 250 | 314.3 (1.00x) | 326.4 (1.04x) | 1250.6 (3.98x) | 308.9 (0.98x) |
| 1000 | 1257.4 (1.00x) | 877.5 (0.70x) | 1365.3 (1.09x) | 774.7 (0.62x) |
| 2000 | 2696.4 (1.00x) | 1897.6 (0.70x) | 2953.7 (1.10x) | 1711.9 (0.63x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 24.6 (1.00x) | 23.8 (0.97x) | 27.3 (1.11x) | 13.5 (0.55x) |
| 250 | 24.9 (1.00x) | 26.4 (1.06x) | 27.6 (1.11x) | 14.0 (0.56x) |
| 1000 | 25.4 (1.00x) | 41.1 (1.62x) | 29.1 (1.15x) | 14.8 (0.58x) |
| 2000 | 25.9 (1.00x) | 61.2 (2.37x) | 28.9 (1.12x) | 15.9 (0.61x) |

### Q4_K_M

Batch: `20261117-160418`.

**Prompt tokens**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 6 | 6 | 5 | 5 |
| 250 | 238 | 238 | 237 | 237 |
| 1000 | 971 | 971 | 970 | 970 |
| 2000 | 1937 | 1937 | 1936 | 1936 |

**TTFT (ms)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 39.8 (1.00x) | 44.1 (1.11x) | 1483.2 (37.29x) | 34.9 (0.88x) |
| 250 | 371.2 (1.00x) | 321.1 (0.87x) | 1493.9 (4.02x) | 450.8 (1.21x) |
| 1000 | 1498.7 (1.00x) | 948.1 (0.63x) | 1605.7 (1.07x) | 1030.3 (0.69x) |
| 2000 | 3168.8 (1.00x) | 2022.0 (0.64x) | 3432.6 (1.08x) | 1917.4 (0.61x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 25.3 (1.00x) | 25.7 (1.02x) | 28.0 (1.10x) | 17.6 (0.69x) |
| 250 | 25.6 (1.00x) | 28.4 (1.11x) | 28.7 (1.12x) | 18.5 (0.72x) |
| 1000 | 26.0 (1.00x) | 43.1 (1.66x) | 29.8 (1.15x) | 19.3 (0.74x) |
| 2000 | 26.6 (1.00x) | 63.3 (2.38x) | 29.7 (1.12x) | 19.9 (0.75x) |

## Qwen3-4B Q4_K_M

Batch: `20261117-173646`; ten measured runs per case. Versions: llama.cpp `20260904-64a155`, OpenVINO `20261111-e65c59`, GenAI `20261026-e2a691`, graph generator `20261117-03349b`. GenAI + OV used GenAI's built-in GGUF reader. All four paths completed at every prompt length.

**Prompt tokens**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 5 | 5 | 5 | 5 |
| 250 | 237 | 237 | 237 | 237 |
| 1000 | 970 | 970 | 970 | 970 |
| 2000 | 1946 | 1946 | 1946 | 1946 |

**TTFT (ms)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 117.2 (1.00x) | 143.9 (1.23x) | 5428.6 (46.31x) | 102.2 (0.87x) |
| 250 | 1412.2 (1.00x) | 1312.7 (0.93x) | 5461.2 (3.87x) | 1581.1 (1.12x) |
| 1000 | 5864.9 (1.00x) | 3974.3 (0.68x) | 6085.7 (1.04x) | 3689.8 (0.63x) |
| 2000 | 15005.2 (1.00x) | 7763.1 (0.52x) | 15905.1 (1.06x) | 7630.0 (0.51x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 67.0 (1.00x) | 83.2 (1.24x) | 72.0 (1.08x) | 51.7 (0.77x) |
| 250 | 68.1 (1.00x) | 93.4 (1.37x) | 73.9 (1.09x) | 52.9 (0.78x) |
| 1000 | 71.1 (1.00x) | 154.3 (2.17x) | 77.5 (1.09x) | 56.3 (0.79x) |
| 2000 | 75.4 (1.00x) | 249.8 (3.32x) | 79.3 (1.05x) | 60.6 (0.80x) |

## Qwen3.5-4B Q4_K_M

Batch: `20261119-000723`; five measured runs per completed case. Versions: llama.cpp `20260904-64a155`, OpenVINO `20261118-aee876`, GenAI `20261118-a057a0`, graph generator `20261117-03349b`. GenAI + OV used the OpenVINO GGUF frontend.

**Prompt tokens**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 5 | 5 | - | 5 |
| 250 | 236 | 236 | - | 237 |
| 1000 | 974 | 974 | - | 992 |
| 2000 | 1956 | 1956 | - | 1990 |

**TTFT (ms)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 136.5 (1.00x) | 179.2 (1.31x) | Exit 2 | 149.5 (1.10x) |
| 250 | 1869.5 (1.00x) | 1667.9 (0.89x) | Exit 2 | 1217.6 (0.65x) |
| 1000 | 7256.5 (1.00x) | 6224.2 (0.86x) | Exit 2 | 3449.0 (0.48x) |
| 2000 | 14973.3 (1.00x) | 9255.3 (0.62x) | Exit 2 | 7052.5 (0.47x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 80.9 (1.00x) | 108.0 (1.34x) | Exit 2 | 67.3 (0.83x) |
| 250 | 81.3 (1.00x) | 109.6 (1.35x) | Exit 2 | 68.0 (0.84x) |
| 1000 | 82.0 (1.00x) | 123.2 (1.50x) | Exit 2 | 68.4 (0.83x) |
| 2000 | 82.7 (1.00x) | 148.9 (1.80x) | Exit 2 | 70.3 (0.85x) |

- **GenAI + Vulkan, no timing:** GGML rejected the 1024-token graph before inference for every prompt. The runner accepts only one-token graphs for recurrent models because padded rows would update recurrent state. A one-token-only configuration was not measured in this batch.
- **GenAI + OV, timed with output errors:** The saved text samples were garbled for all four prompt cases. llama.cpp + OV also reported non-identical output across the five runs of the shortest prompt.

## Gemma-4-12B-IT Q4_K_M

Batch: `20261118-225030`; five measured runs per case. Versions: llama.cpp `20260904-64a155`, OpenVINO `20261118-aee876`, GenAI `20261118-a057a0`, graph generator `20261117-03349b`. GenAI + OV used the OpenVINO GGUF frontend.

**Prompt tokens**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 6 | 6 | 6 | 6 |
| 250 | 233 | 233 | 233 | 233 |
| 1000 | 964 | 964 | 982 | 982 |
| 2000 | 1934 | 1934 | 1970 | 1970 |

**TTFT (ms)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 369.9 (1.00x) | 428.2 (1.16x) | 16280.4 (44.01x) | 305.9 (0.83x) |
| 250 | 4069.4 (1.00x) | 3720.4 (0.91x) | 16319.2 (4.01x) | 2872.0 (0.71x) |
| 1000 | 17403.5 (1.00x) | 9436.0 (0.54x) | 18123.1 (1.04x) | 9198.1 (0.53x) |
| 2000 | 38490.9 (1.00x) | 19392.4 (0.50x) | 42037.2 (1.09x) | 18295.8 (0.48x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 191.2 (1.00x) | 276.5 (1.45x) | 196.6 (1.03x) | 137.2 (0.72x) |
| 250 | 193.1 (1.00x) | 273.5 (1.42x) | 201.7 (1.04x) | 141.5 (0.73x) |
| 1000 | 197.7 (1.00x) | 324.2 (1.64x) | 215.3 (1.09x) | 148.0 (0.75x) |
| 2000 | 203.9 (1.00x) | 367.3 (1.80x) | 217.7 (1.07x) | 167.9 (0.82x) |

All four paths completed at every prompt length. **llama.cpp + OV, timed with output errors:** For prompt 2000, the saved sample contained repeated channel markers and digits, and output varied across runs.