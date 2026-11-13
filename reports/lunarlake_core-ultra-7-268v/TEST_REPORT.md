# Test report: GGML device on an Intel Core Ultra 7 268V (Lunar Lake, Arc 140V)

This report analyzes one run of the test package (`scripts\run_tests.py`, all 34 models). The raw output is [summary.md](summary.md) in this folder. The [gap analysis](../../docs/GAP_ANALYSIS.md) relates these results to the design review's findings.

## Summary

- **Coverage.** The GGML device ran all 34 models on the CPU and 32 on the GPU. GenAI's own GGUF path (OV CPU and OV GPU) can load only 16 of them.
- **Decode, GPU against GPU.** On the same Arc 140V, the GGML device on ggml's Vulkan backend decodes 1.17× faster than OpenVINO's own GPU plugin. That's the geometric mean over the 16 models both paths ran. The lead grows with model size: about equal below 1.5 GiB, 1.30× for the 7–8B models.
- **Decode against the CPU.** The GGML GPU path decodes 2.2× faster than OpenVINO's CPU plugin. Even ggml's CPU backend, though limited to 4 threads, is 1.6× faster than OpenVINO's CPU plugin.
- **Prefill is the weak spot.** OpenVINO's GPU plugin reaches the first token 1.7× faster than the GGML GPU path with its 8-token prefill bucket. Without the bucket, the GGML path is another 3.7× slower.
- **One GPU failure.** Gemma-4 E2B and E4B fail on the GPU during load, without an error message. The likely cause is a per-buffer size limit in Intel's Vulkan driver, but that needs the run's logs to confirm.
- **Output agreement.** The same graph on ggml's CPU and GPU backends gives identical text for 24 of 32 models. The other 8 differ late in the answer, which is normal float rounding.

## Test setup

| | |
| :-- | :-- |
| OS | Windows 11 (10.0.26200), run over Remote Desktop |
| CPU | Intel Core Ultra 7 268V (Lunar Lake): 4 P-cores + 4 LP E-cores, 8 threads |
| Memory | 32 GB, shared by CPU and GPU. Per Intel's specifications, LPDDR5X-8533 on package, about 136 GB/s peak. |
| GPU | Intel Arc 140V (Xe2), integrated; driver 32.0.101.8425, Vulkan 1.4.333 |
| OpenVINO devices | `CPU`, `GGML`, `GPU` (the Arc 140V); the OV GPU baseline ran on `GPU` |
| Package | `ggml_ov_test` as built on 2026-11-12 (see `VERSIONS.txt`) |
| Procedure | for every model: one untimed warm-up generation, then greedy decoding of up to 32 new tokens for the prompt "The capital of France is" in the model's chat template |

The system information identifies the CPU as a Core Ultra 7 268V, which is Lunar Lake (Xe2 graphics), not Panther Lake (Xe3).

## Results

### Coverage

| Path | Ran | Of | Not run, and why |
| :-- | --: | --: | :-- |
| GGML CPU, 1-token graph (`d1`) | 34 | 34 | |
| GGML CPU, 8-token bucket (`b8`) | 30 | 30 | the 4 Qwen3.5 models have recurrent layers, and buckets are rejected for them by design |
| GGML GPU, `d1` | 32 | 34 | Gemma-4 E2B and E4B fail at load (see below) |
| GGML GPU, `b8` | 28 | 30 | the same two Gemma-4 models |
| OV CPU | 16 | 34 | GenAI's GGUF path rejects 18 models: architectures gemma3, granite, exaone, hunyuan-dense; tensor types in gemma4, Phi, Bonsai, Qwen3.5 |
| OV GPU | 16 | 34 | the same 18 models |

Every family in the validated list runs on the GGML device, including the 18 models OpenVINO's own GGUF path can't load. gemma-4-12B (7.1 GiB) also runs on this GPU, because it can use the shared 32 GB.

### Decode speed

Time per generated token (TPOT) in ms, for representative models:

| Model | GiB | GGML CPU | GGML GPU | OV CPU | OV GPU |
| :-- | --: | --: | --: | --: | --: |
| Llama-3.2-1B | 0.75 | 15.6 | 13.6 | 26.2 | 14.7 |
| Qwen3-4B | 2.33 | 46.9 | 34.4 | 71.7 | 36.8 |
| Mistral-7B v0.3 | 4.07 | 77.4 | 49.2 | 131.1 | 63.6 |
| Llama-3.1-8B | 4.58 | 82.0 | 56.6 | 143.5 | 69.3 |
| Qwen3-8B | 4.68 | 84.0 | 54.7 | 139.6 | 73.4 |

How much faster the GGML GPU path decodes than OV GPU (OV GPU TPOT divided by GGML GPU TPOT), by model size:

| Model size | Models | Geometric mean | Range |
| :-- | --: | --: | :-- |
| below 1.5 GiB | 5 | 1.03× | 0.85×–1.19× |
| 1.5–3 GiB | 3 | 1.11× | 1.07×–1.15× |
| 3–5 GiB | 8 | 1.30× | 1.22×–1.38× |
| all | 16 | 1.17× | |

Decode is limited by memory bandwidth, so the bytes of weights read per second show how close each path gets to the hardware. For the 8B models, dividing the GGUF size by TPOT gives the figures below. They slightly overestimate the real reads, because only one row of the token-embedding table is read per token.

| GGML GPU | OV GPU | GGML CPU | OV CPU |
| :-- | :-- | :-- | :-- |
| 87–92 GB/s (about 65% of peak) | 68–71 GB/s | 60 GB/s | 34–36 GB/s |

Other observations:

- **GGML CPU is faster than OV CPU** (1.64× geometric mean), even though the loader leaves ggml's CPU backend at its default of 4 threads. This run doesn't show why. Possible causes are OpenVINO's CPU defaults (u8 KV cache, dynamic int8 activations), thread placement on the LP E-cores, or the kernels GenAI's GGUF path uses for these quantization types.
- **The smallest models decode slower on the GPU than on the CPU:** granite-4.0-350m takes 8.2 ms on the GPU against 6.2 ms on the CPU, and Qwen3-0.6B 10.9 against 9.5. Both GPU paths seem to have a fixed cost of about 8–10 ms per token on this machine (OV GPU: 9.3 ms for Qwen3-0.6B), which dominates for tiny models. The Remote Desktop session may add to this; a local run would tell.
- **GGML GPU against GGML CPU:** 1.34× geometric mean over the 32 models.

### Prefill speed (time to first token)

| Model | Prompt tokens | GGML GPU `d1` | GGML GPU `b8` | OV GPU | OV CPU |
| :-- | --: | --: | --: | --: | --: |
| Llama-3.2-1B | 40 | 470 | 93 | 114 | 48 |
| Qwen3-8B | 13 | 674 | 213 | 91 | 341 |
| Llama-3.1-8B | 40 | 2125 | 524 | 205 | 284 |
| granite-3.2-8b | 65 | 3613 | 979 | | |

All times are in ms. The granite model has no OV results because GenAI's GGUF path can't load it.

- OV GPU reaches the first token 1.74× faster than GGML GPU `b8` (geometric mean, 16 models). GGML `b8` is faster only for Llama-3.2-1B.
- Without the bucket, GGML GPU prefill is 3.7× slower again, because it processes one prompt token per graph run.
- The cause is design finding L7. The artifacts take at most 8 rows per graph run, while OpenVINO processes the whole prompt in one call. Buckets of 16 tokens or more need artifacts dumped with a CPU-only llama.cpp, which the package's dump setup already uses, so larger buckets can now be tried.

### Output agreement

| Comparison | Same text | Different |
| :-- | --: | --: |
| GGML GPU against GGML CPU (same graph) | 24 | 8 |
| GGML CPU against OV CPU | 8 | 8 |
| GGML GPU against OV GPU | 5 | 11 |

- The 8 GGML CPU/GPU differences are granite-4.0-350m, granite-4.0-micro, Qwen3.5-0.8B, Qwen3-4B, Phi-3-mini, Phi-3.5-mini, Phi-4-mini and Bonsai-8B. The answers diverge on a near-tie token, as expected when the GPU accumulates in different precision.
- The Phi models also run past the end of their answer, because GenAI's GGUF path doesn't stop at their `<|end|>` token. Longer outputs give more chances to diverge.
- The GGML/OV differences are expected, because OpenVINO's CPU and GPU plugins use lower-precision KV caches and activations by default. The summary records only whether the texts match, not where they first differ; `cpu\e2e.md` and `gpu\e2e.md` of the run show the texts.
- This run checks agreement, not accuracy. An accuracy comparison would need a reference such as llama.cpp's own output, or a benchmark.

## Failures and anomalies

### Gemma-4 E2B and E4B fail on the GPU

Both models fail at load, in the `d1` and `b8` runs alike. The run's output ends right after ggml lists the Vulkan devices, and there's no error message.

In `summary.md`, the Notes cell for these rows is garbled, because the listing contains `|` characters. The package's `summarize.py` now reports such failures as "failed: access violation, no error message" or with the exit code instead.

The same artifacts work on this machine's CPU (33.3 and 58.8 ms per token), and gemma-4-12B works on this GPU.

What separates E2B and E4B from the 12B is the per-layer embedding table, which is a single very large tensor:

| Model | Largest tensor | Size |
| :-- | :-- | --: |
| gemma-4-E2B | `per_layer_token_embd.weight` (8960 × 262144, Q6_K) | 1.79 GiB |
| gemma-4-E4B | `per_layer_token_embd.weight` (10752 × 262144, Q6_K) | 2.15 GiB |
| gemma-4-12B (works) | `token_embd.weight` (3840 × 262144, Q6_K) | 0.77 GiB |

The likely explanation is that a tensor this large exceeds a per-buffer limit of Intel's Vulkan driver (`maxStorageBufferRange`, `maxMemoryAllocationSize` or `maxBufferSize`), and the failure goes unchecked. That's a hypothesis. Two things would confirm or refute it:

1. the logs of these runs, `results\<run>\gpu\google_gemma-4-E2B-it-Q4_K_M.GGML_Vulkan0_d1.log` and the E4B one;
2. the driver's limits: `vulkaninfo | findstr /i "maxStorageBufferRange maxMemoryAllocationSize maxBufferSize"`.

If a size limit is the cause, ggml would need to split the table, or the plugin should keep this tensor on the CPU and report that it did.

### The GPU is listed twice

`vulkaninfo` lists the Arc 140V twice (GPU0 and GPU1, same driver 101.8425), so ggml sees `Vulkan0` and `Vulkan1`. The tests used `Vulkan0`. This doesn't affect the results, but it usually means the Vulkan driver is registered twice. It's worth checking `HKLM\SOFTWARE\Khronos\Vulkan\Drivers`, because a stale registration could make ggml pick an old driver.

## Conclusions

- On Intel's own GPU, the GGML device (llama.cpp's Vulkan kernels behind OpenVINO GenAI) works and is competitive. It decodes 1.17× faster than OpenVINO's GPU plugin overall and 1.30× faster at 7–8B, and it runs twice as many of the validated models (32 against 16).
- Prefill is the main gap. OpenVINO's GPU plugin is 1.7× faster to the first token, because the GGML artifacts top out at 8-token batches.
- The only GPU failures are the Gemma-4 per-layer-embedding models, which need their logs.

## Next steps

1. Collect the Gemma-4 E2B/E4B logs and the Vulkan limits (above) to confirm the cause.
2. Dump 16-, 32- and 64-token prefill buckets and rerun the first-token measurements.
3. Repeat the run on a Panther Lake machine (Xe3), and locally rather than over Remote Desktop, to check the per-token GPU overhead.
4. Let the loader use more CPU threads, instead of ggml's default of 4, and remeasure GGML CPU.
5. Make GenAI's GGUF path stop at end-of-turn tokens, so that the Phi and Gemma-4 answers end where they should.
6. Test long prompts, which exercise sliding-window attention (design finding L1), and the paged pipeline on the GPU.
