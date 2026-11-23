# Performance notes: Four GGUF paths on Intel integrated graphics

Six Q4_K_M models were measured on 2026-11-20. Batch IDs identify the source summaries in `llama-work-note/results/compare-gpu-4paths`.

## Tested models

- [Llama-3.2-1B-Instruct Q4_K_M](#llama-32-1b-instruct-q4_k_m)
- [Qwen3-4B Q4_K_M](#qwen3-4b-q4_k_m)
- [Qwen3.5-4B Q4_K_M](#qwen35-4b-q4_k_m)
- [Gemma-4-E2B-IT Q4_K_M](#gemma-4-e2b-it-q4_k_m)
- [Gemma-4-E4B-IT Q4_K_M](#gemma-4-e4b-it-q4_k_m)
- [Gemma-4-12B-IT Q4_K_M](#gemma-4-12b-it-q4_k_m)

## Shared configuration

All runs target the Intel integrated GPU on a Windows system with a Core Ultra 9 285K. Each completed case has **1 warm-up and 5 measured runs**. Each run requests **32 output tokens**, uses greedy decoding, ignores EOS, and applies no chat template.

| Path | Device | Model input | Context and execution settings |
| --- | --- | --- | --- |
| llama.cpp + Vulkan | `Vulkan1` | GGUF via llama.cpp | 4096-token context; `-ub 1024`; flash attention and f16 KV requested |
| llama.cpp + OV | `GPU.0` | GGUF via llama.cpp | 4096-token context; `-ub 1024`; flash attention and f16 KV requested |
| GenAI + Vulkan | `Vulkan1` | Exported GGML graphs via the GGML device | `n_kv=4096`; 1024-token prefill bucket; 1-token decode graph; flash attention and KV settings encoded in the graphs |
| GenAI + OV | `GPU.0` | OpenVINO GGUF frontend (`ovfe`) | Out-of-order GPU queue; effective KV capacity, KV precision, and flash attention settings were not verified |

**Builds:** llama.cpp `20260904-64a155`, OpenVINO `20261120-006aa1`, GenAI `20261119-ce675e`, graph generator `20261119-9856f8`.

**Reading the tables:** Values are medians of the measured runs. **TTFT (ms)** is prompt inference time up to the first token's logits; **TPOT (ms/token)** is inference time per subsequent token. The timings use llama.cpp's prompt/eval times and GenAI's raw inference durations, excluding model loading, sampling, and pipeline overhead. Ratios use llama.cpp + Vulkan as `1.00x` for the same model, prompt, and metric. Prompt labels are nominal; the prompt-token tables show actual counts, with `-` indicating no recorded count.

## Short-prompt bucket sensitivity

On 2026-11-23, only **GenAI + Vulkan** was rerun on `Vulkan1` with the same builds and `n_kv=4096`. Within each model, all five configurations used identical input token IDs for `The capital of France is`. Values are inference TTFT medians in milliseconds after one warm-up and five measured runs per configuration.

| Model (Q4_K_M) | Input tokens | `b1` only | `b1+b8` | `b1+b32` | `b1+b1024` | `b1+b8+b1024` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct | 6 | 155.2 | 32.2 | 114.9 | 1351.9 | 32.0 |
| Qwen3-4B | 5 | 340.0 | 94.1 | 395.0 | 4909.3 | 92.1 |

Adding `b8` alongside `b1024` brought short-prompt TTFT close to the `b1+b8` result for both models. With both buckets configured, the selection rule picks `b8` for these prompts, padding 2 rows for Llama and 3 for Qwen; `b1024` would pad 1018 and 1019 rows, respectively. These selections are inferred from the code, not directly logged. Within each model, the 32 output token IDs logged for the final measured run matched across all five configurations.

The roughly 54x and 69x short-prompt TTFT ratios in the four-path tables below apply to the original `b1+b1024` configuration. This follow-up does not update those ratios or measure longer prompts.

## Llama-3.2-1B-Instruct Q4_K_M

Batch: `20261120-201355`. All four paths completed.

- **GenAI + OV:** The saved sample at prompt 5 consisted of exclamation marks; the other three samples were garbled.
- **llama.cpp + OV:** At prompt 250, the generated text was not identical across the five measured runs.

**Prompt tokens**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 6 | 6 | 6 | 6 |
| 250 | 238 | 238 | 238 | 238 |
| 1000 | 971 | 971 | 971 | 971 |
| 2000 | 1937 | 1937 | 1937 | 1937 |

**TTFT (ms)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 25.2 (1.00x) | 44.9 (1.78x) | 1358.7 (53.98x) | 40.6 (1.61x) |
| 250 | 336.0 (1.00x) | 318.4 (0.95x) | 1362.7 (4.06x) | 289.5 (0.86x) |
| 1000 | 1338.6 (1.00x) | 918.0 (0.69x) | 1451.5 (1.08x) | 880.9 (0.66x) |
| 2000 | 2804.6 (1.00x) | 2222.0 (0.79x) | 3073.0 (1.10x) | 1917.4 (0.68x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 24.6 (1.00x) | 22.9 (0.93x) | 27.8 (1.13x) | 17.8 (0.72x) |
| 250 | 25.0 (1.00x) | 25.9 (1.04x) | 28.4 (1.14x) | 18.4 (0.74x) |
| 1000 | 26.0 (1.00x) | 41.8 (1.60x) | 30.5 (1.17x) | 19.4 (0.75x) |
| 2000 | 26.9 (1.00x) | 63.9 (2.37x) | 30.3 (1.12x) | 20.8 (0.77x) |

## Qwen3-4B Q4_K_M

Batch: `20261120-161106`. All four paths completed.

- **GenAI + OV:** The saved sample at prompt 5 consisted of exclamation marks; the other three samples contained repetition or garbled text.
- **llama.cpp + OV:** At prompt 5, the saved sample repeatedly said "The capital of Paris is Paris." At prompt 1000, the generated text was not identical across the five measured runs.

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
| 5 | 71.1 (1.00x) | 130.7 (1.84x) | 4916.4 (69.16x) | 109.5 (1.54x) |
| 250 | 1291.3 (1.00x) | 1410.1 (1.09x) | 4960.6 (3.84x) | 1060.6 (0.82x) |
| 1000 | 5363.3 (1.00x) | 3729.9 (0.70x) | 5574.0 (1.04x) | 3155.3 (0.59x) |
| 2000 | 14137.1 (1.00x) | 9411.8 (0.67x) | 14857.3 (1.05x) | 6983.6 (0.49x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 66.6 (1.00x) | 81.4 (1.22x) | 72.1 (1.08x) | 51.7 (0.78x) |
| 250 | 66.4 (1.00x) | 92.5 (1.39x) | 78.1 (1.18x) | 53.2 (0.80x) |
| 1000 | 73.9 (1.00x) | 157.0 (2.12x) | 96.2 (1.30x) | 57.7 (0.78x) |
| 2000 | 81.3 (1.00x) | 253.3 (3.12x) | 93.9 (1.16x) | 63.3 (0.78x) |

## Qwen3.5-4B Q4_K_M

Batch: `20261120-153933`.

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
| 5 | 86.6 (1.00x) | 173.9 (2.01x) | Exit 2 | 155.3 (1.79x) |
| 250 | 1734.4 (1.00x) | 1626.9 (0.94x) | Exit 2 | 1180.8 (0.68x) |
| 1000 | 6852.6 (1.00x) | 6135.9 (0.90x) | Exit 2 | 3429.3 (0.50x) |
| 2000 | 14216.0 (1.00x) | 9130.6 (0.64x) | Exit 2 | 7051.5 (0.50x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 79.5 (1.00x) | 105.7 (1.33x) | Exit 2 | 70.6 (0.89x) |
| 250 | 77.7 (1.00x) | 107.6 (1.38x) | Exit 2 | 66.6 (0.86x) |
| 1000 | 79.9 (1.00x) | 121.7 (1.52x) | Exit 2 | 67.4 (0.84x) |
| 2000 | 80.7 (1.00x) | 148.0 (1.83x) | Exit 2 | 68.6 (0.85x) |

- **GenAI + Vulkan, no timing or text:** All four prompts exited with code `2`. The GGML runner rejected the 1024-row prefill graph because padding would update recurrent state. No one-token-only configuration was measured.
- **GenAI + OV, timed with output errors:** The saved text samples were garbled at all four prompt lengths.
- **llama.cpp + OV:** At prompt 5, the generated text was not identical across the five measured runs.

## Gemma-4-E2B-IT Q4_K_M

Batch: `20261120-170550`. Both Vulkan paths completed. Neither OpenVINO path returned timing or generated text at any prompt length.

- **llama.cpp + OV:** Exited at the start of generation with Windows status `0xC0000005` (`-1073741819`, access violation).
- **GenAI + OV:** Exited with code `2` after `clBuildProgram` returned `CL_BUILD_PROGRAM_FAILURE`.

**Prompt tokens**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 6 | - | 6 | - |
| 250 | 233 | - | 233 | - |
| 1000 | 964 | - | 982 | - |
| 2000 | 1934 | - | 1970 | - |

**TTFT (ms)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 56.2 (1.00x) | Exit -1073741819 | 2871.2 (51.10x) | Exit 2 |
| 250 | 791.6 (1.00x) | Exit -1073741819 | 2883.0 (3.64x) | Exit 2 |
| 1000 | 3089.4 (1.00x) | Exit -1073741819 | 3198.7 (1.04x) | Exit 2 |
| 2000 | 7005.2 (1.00x) | Exit -1073741819 | 7434.1 (1.06x) | Exit 2 |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 47.2 (1.00x) | Exit -1073741819 | 51.2 (1.08x) | Exit 2 |
| 250 | 47.7 (1.00x) | Exit -1073741819 | 57.7 (1.21x) | Exit 2 |
| 1000 | 50.9 (1.00x) | Exit -1073741819 | 57.7 (1.13x) | Exit 2 |
| 2000 | 54.4 (1.00x) | Exit -1073741819 | 59.1 (1.09x) | Exit 2 |

## Gemma-4-E4B-IT Q4_K_M

Batch: `20261120-172350`. Both Vulkan paths completed. Neither OpenVINO path returned timing or generated text; the failures matched [Gemma E2B](#gemma-4-e2b-it-q4_k_m).

**Prompt tokens**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 6 | - | 6 | - |
| 250 | 233 | - | 233 | - |
| 1000 | 964 | - | 982 | - |
| 2000 | 1934 | - | 1970 | - |

**TTFT (ms)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 94.0 (1.00x) | Exit -1073741819 | 5708.1 (60.71x) | Exit 2 |
| 250 | 1522.6 (1.00x) | Exit -1073741819 | 5738.2 (3.77x) | Exit 2 |
| 1000 | 6080.0 (1.00x) | Exit -1073741819 | 6252.2 (1.03x) | Exit 2 |
| 2000 | 12779.4 (1.00x) | Exit -1073741819 | 13415.3 (1.05x) | Exit 2 |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 79.2 (1.00x) | Exit -1073741819 | 86.0 (1.09x) | Exit 2 |
| 250 | 81.1 (1.00x) | Exit -1073741819 | 92.9 (1.14x) | Exit 2 |
| 1000 | 83.1 (1.00x) | Exit -1073741819 | 93.0 (1.12x) | Exit 2 |
| 2000 | 84.9 (1.00x) | Exit -1073741819 | 92.9 (1.09x) | Exit 2 |

## Gemma-4-12B-IT Q4_K_M

Batch: `20261120-174348`.

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
| 5 | 197.5 (1.00x) | 410.2 (2.08x) | 14319.4 (72.49x) | 306.8 (1.55x) |
| 250 | 3687.7 (1.00x) | 3686.6 (1.00x) | 14416.4 (3.91x) | 2873.3 (0.78x) |
| 1000 | 15672.8 (1.00x) | 9247.7 (0.59x) | 16164.2 (1.03x) | 9367.6 (0.60x) |
| 2000 | 34627.1 (1.00x) | 18900.7 (0.55x) | 36409.0 (1.05x) | 18629.1 (0.54x) |

**TPOT (ms/token)**

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 185.9 (1.00x) | 285.9 (1.54x) | 192.0 (1.03x) | 138.5 (0.74x) |
| 250 | 187.6 (1.00x) | 290.7 (1.55x) | 193.9 (1.03x) | 138.9 (0.74x) |
| 1000 | 192.8 (1.00x) | 343.6 (1.78x) | 209.2 (1.09x) | 147.6 (0.77x) |
| 2000 | 198.0 (1.00x) | 385.4 (1.95x) | 209.4 (1.06x) | 165.8 (0.84x) |

All four paths completed at every prompt length. **llama.cpp + OV, timed with output errors:** The prompt 2000 sample contained repeated channel markers and digits, and the generated text was not identical across the five measured runs.