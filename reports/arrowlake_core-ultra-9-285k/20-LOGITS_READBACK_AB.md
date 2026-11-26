# Asynchronous logits readback on Intel Core Ultra 9 285K (Arrow Lake)

This report records the A/B measurements behind [finding 20](../../docs/GAP_ANALYSIS_PERF.md#2-new-gaps-found-by-testing): the GGML device reads logits synchronously, which on a UMA GPU makes the CPU copy them from uncached Vulkan memory.

- **Result:** On the Intel iGPU (`Vulkan1`), the asynchronous read lowered Llama-3.2-1B decode TPOT by **1.79-2.20 ms/token (6.0%-7.3%)** in both run orders. The 32 output token IDs matched.
- **Larger vocabularies:** Qwen3-4B and Phi-4-mini were also faster in both run orders, with matching token IDs. The size of the gain varied with run order: 1.39-6.55 ms/token for Qwen3-4B and 2.80-5.39 ms/token for Phi-4-mini.
- **Non-UMA control:** On RTX 5090 (`Vulkan0`), TPOT did not change measurably.
- **Status:** Submitted in [ynimmaga/openvino #118](https://github.com/ynimmaga/openvino/pull/118) as commit `099d14d` on branch `ggml_plugin_async`.

## Change under test

Only `GgmlModel::read_logits` in `src/ggml_cgraph_loader/src/model.cpp` differs between the two builds. The baseline calls `ggml_backend_tensor_get(m_impl->out.last, ...)`. The test build reads through the backend that owns the logits tensor:

```cpp
void GgmlModel::read_logits(float* out) const {
    // With a scheduler the logits may live on the CPU fallback backend; read through the one that owns them.
    ggml_backend_t backend =
        m_impl->sched ? ggml_backend_sched_get_tensor_backend(m_impl->sched, m_impl->out.last) : m_impl->backend;
    if (!backend) {
        ggml_backend_tensor_get(m_impl->out.last, out, 0, logits_size() * sizeof(float));
        return;
    }
    ggml_backend_tensor_get_async(backend, m_impl->out.last, out, 0, logits_size() * sizeof(float));
    ggml_backend_synchronize(backend);
}
```

Why it helps, from reading ggml `v0.23.0` (the version fetched by OpenVINO `thirdparty/ggml`):

- **Synchronous path:** `ggml_backend_tensor_get` reaches `ggml_vk_buffer_read_2d`. On a UMA device with a host-visible buffer, it waits on a fence and runs `memcpy` from the mapped buffer. On UMA, `ggml_vk_create_buffer_device` allocates graph buffers as `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` without requesting `HOST_CACHED`.
- **Asynchronous path:** `ggml_backend_vk_get_tensor_2d_async` does not take the UMA branch. When the destination is not pinned memory, it records a GPU copy into the context's `sync_staging` buffer, which prefers `HOST_CACHED`. It then synchronizes, and the CPU copies from the staging buffer.
- llama.cpp reads logits the same way, with `ggml_backend_tensor_get_async` in `llama_context`.

The scheduler branch selects the backend that owns the logits. A fixed Vulkan backend would fail an assertion in `ggml_backend_vk_get_tensor_2d_async` if the logits were allocated on the CPU fallback. This branch was not exercised in the decode steps measured here: for all three models, the logs report `backend: Vulkan1`. This value comes from `GgmlModel::backend_name` of the main `b1` graph, which appends `(CPU for ...)` when any op falls back to the CPU.

## Configuration

| Item | Value |
| --- | --- |
| System | Windows, Intel Core Ultra 9 285K; Intel iGPU as `Vulkan1` (`uma: 1`), RTX 5090 as `Vulkan0` (`uma: 0`) |
| Models | Q4_K_M GGUF: Llama-3.2-1B-Instruct, Qwen3-4B, Phi-4-mini-instruct |
| Graphs | Generator `20261119-9856f8`: `b1/n4096` main graph plus the prefill buckets listed in the results table |
| Builds | OpenVINO `20261120-006aa1` (baseline), the same commit plus the change above (test; same `model.cpp` as commit `099d14d`), GenAI `20261119-ce675e` |
| Prompt | [prompt-1000.txt](./prompts/prompt-1000.txt); actual token counts are in the results table |
| Runs | 32 output tokens, greedy decoding, EOS ignored, no chat template; each build in its own process, 1 warm-up and 5 measured runs |
| Metric | Decode TPOT: median of GenAI's raw per-token inference durations (`infer:` in the logs) |

Bytes read per step are the vocabulary size times 4 (FP32 logits), taken from the `result_output` node of each `b1` graph. Each A/B pair was measured in both run orders.

## Results on the Intel iGPU (`Vulkan1`)

| Model | Bytes per step | Gain, baseline first | Gain, test first |
| --- | ---: | ---: | ---: |
| Llama-3.2-1B | 513,024 | 1.79 ms/token (6.0%) | 2.20 ms/token (7.3%) |
| Qwen3-4B | 607,744 | 6.55 ms/token (6.5%) | 1.39 ms/token (1.5%) |
| Phi-4-mini | 800,256 | 5.39 ms/token (6.9%) | 2.80 ms/token (3.7%) |

| Model | Prompt tokens | Prefill buckets | Run order | Build | Run ID | TPOT, median (ms/token) | TPOT, measured runs (ms/token) |
| --- | ---: | --- | --- | --- | --- | ---: | --- |
| Llama-3.2-1B | 971 | `b8`, `b1024` | Baseline first | Baseline | `20261126-160258` | 29.96 | 29.96, 29.88, 30.03, 29.81, 30.01 |
| | | | | Asynchronous read | `20261126-160407` | **28.17** | 28.11, 28.17, 28.19, 28.10, 28.21 |
| | | | Test first | Asynchronous read | `20261126-162057` | **28.24** | 28.05, 28.40, 28.24, 28.50, 27.99 |
| | | | | Baseline | `20261126-162224` | 30.45 | 30.29, 30.55, 30.45, 30.49, 30.40 |
| Qwen3-4B | 970 | `b8`, `b1024` | Baseline first | Baseline | `20261126-162617` | 101.36 | 101.67, 101.27, 101.98, 101.36, 101.09 |
| | | | | Asynchronous read | `20261126-162746` | **94.81** | 94.81, 95.07, 94.13, 95.11, 92.77 |
| | | | Test first | Asynchronous read | `20261126-162859` | **93.72** | 92.18, 93.36, 93.72, 94.87, 94.16 |
| | | | | Baseline | `20261126-163013` | 95.11 | 96.07, 94.40, 95.11, 95.45, 94.97 |
| Phi-4-mini | 991 | `b1024` | Baseline first | Baseline | `20261126-163352` | 77.67 | 75.69, 76.83, 77.67, 77.80, 77.78 |
| | | | | Asynchronous read | `20261126-163512` | **72.28** | 72.36, 72.28, 72.28, 72.12, 72.15 |
| | | | Test first | Asynchronous read | `20261126-163649` | **73.61** | 73.25, 72.78, 74.28, 74.15, 73.61 |
| | | | | Baseline | `20261126-163806` | 76.42 | 76.42, 75.24, 76.20, 77.22, 76.77 |

- **Direction:** The asynchronous read was faster in all six pairs. For each model, all four runs generated the same 32 token IDs.
- **Magnitude:** Stable only for Llama-3.2-1B, where the two builds' ranges do not overlap in either order. The two Qwen3-4B baseline medians differ by 6.25 ms/token, more than its smaller gain, and its test-first ranges overlap (94.40-94.87 ms/token). The Phi-4-mini ranges do not overlap, but its gain still differs by 2.6 ms/token between orders.
- Phi-4-mini used only the `b1024` prefill bucket because no `b8` graph was exported for it. Prefill buckets do not affect decode.

Llama-3.2-1B output token IDs, identical across its four runs:

```text
578 3925 315 10873 374 539 1664 1773 37784 11 323 279 3488 315 1405 264 8448 574 389 279 1825 9581 1047 912 24473 4320 13 58095 1105 1511 264 10824
```

## Control on RTX 5090 (`Vulkan0`)

Llama-3.2-1B only, run IDs `20261126-155002` (baseline) and `20261126-155036` (test). These runs used only the `b1/n4096` graph, with no prefill buckets, and 1 warm-up and 3 measured runs.

| Metric | Baseline | Asynchronous read |
| --- | ---: | ---: |
| TPOT, median (ms/token) | 1.34 | 1.37 |

Both builds generated the same token IDs. No gain is expected here: on a non-UMA device, the synchronous read already copies through a staging buffer on the transfer queue.

## Open items

- Repeat interleaved runs (for example, baseline and test alternating three times each) to estimate the gain for Qwen3-4B and Phi-4-mini more precisely.
- Exercise the scheduler branch with a model that needs a CPU fallback op.
