# GGML device on OpenVINO: CPU and GPU test package

This package runs GGUF models through OpenVINO GenAI's `LLMPipeline` in up to four ways and compares them:

| Name in the results | What runs |
| :-- | :-- |
| **GGML CPU** | OpenVINO's experimental `GGML` device replays llama.cpp's compute graph on ggml's CPU backend |
| **GGML GPU** | the same graph on ggml's Vulkan backend (for example, on an Intel Arc GPU) |
| **OV CPU** | GenAI's own GGUF support on OpenVINO's CPU plugin (the CPU baseline) |
| **OV GPU** | GenAI's own GGUF support on OpenVINO's GPU plugin (the GPU baseline). It runs only when OpenVINO lists an Intel GPU. |

The models are the Q4_K_M files from the "Validated Models" table of the llama.cpp OpenVINO backend: 34 text decoders. `VERSIONS.txt` lists the exact commits and build options. `BUILD.md` explains how to rebuild everything from source on Windows: OpenVINO, GenAI, the samples, and the artifact dumper.

## Results and documents

- [Test report](reports/lunarlake_core-ultra-7-268v/TEST_REPORT.md) from a Core Ultra 7 268V with an Arc 140V (Lunar Lake). The GGML device ran all 34 models on the CPU and 32 on the GPU, and decoded 1.17× faster than OpenVINO's GPU plugin on the same GPU. Prefill is its main gap.
- [Gap analysis](docs/GAP_ANALYSIS.md): the design review's findings after these tests, and the new gaps the tests found.
- [Performance gap analysis](docs/GAP_ANALYSIS_PERF.md): NPU GGUF performance, the GenAI BOS fix, and additional SWA and large-bucket verification.
- [Performance notes](reports/arrowlake_core-ultra-9-285k/PERFORMANCE_NOTES.md) from a Core Ultra 9 285K (Arrow Lake): latest four-path measurements on its integrated graphics, grouped by model and quantization.
- [Architecture and design review](docs/architecture.pdf) of the GGML device, its GenAI integration and the artifact dumper. It is also in [HTML](docs/architecture.html); download that to view it in a browser.
- [BUILD.md](BUILD.md): how to build everything from source on Windows.
- [PR_GATEKEEPING.md](PR_GATEKEEPING.md): the pull requests to Yamini's repositories that this package needs, and their status.

The GitHub repository doesn't store `bin\`. The release zip `ggml_ov_test.zip`, on the repository's [Releases](https://github.com/taste1981/ov_ggml/releases) page, holds the whole package, `bin\` included.

## Contents

```
bin\          ggml_llm.exe (test driver), hello_query_device.exe, and every DLL it needs: OpenVINO with
              its CPU, GPU and GGML plugins, ggml (CPU, Vulkan), GenAI, tokenizers, TBB, Visual C++ runtime.
              In the release zip only.
artifacts\    <model>\d1.json (1 token per step) and d8.json (8-token prefill bucket), dumped from llama.cpp
models\       manifest.json (the exact GGUF files); downloaded models go here
scripts\      run_tests.py      run the whole test and write a summary
              fetch_models.py   download models from Hugging Face (resumes)
              e2e_llm.py        the per-model runner that run_tests.py uses
              summarize.py      merge the CPU and GPU passes into summary.md
              gguf_meta.py      inspect a GGUF (stop tokens, chat template)
reports\      results of these scripts, with a test report
docs\         gap analysis; architecture and design review (PDF, HTML)
src\ggml_llm\ driver source (C++, CMake)
src\genai_samples\  CMake wrapper that builds GenAI's text-generation samples (see BUILD.md)
BUILD.md      how to build all of this from source
PR_GATEKEEPING.md  the pull requests this package needs
```

## Requirements

- Windows 11 x64. The package was built on Windows Server 2025 and tested on Windows 11.
- An Intel graphics driver. It provides Vulkan, for the GGML GPU runs, and OpenCL, for OpenVINO's GPU plugin. `C:\Windows\System32\vulkaninfo.exe --summary` should list the GPU. The Vulkan SDK is not needed.
- Python 3.8 or later on `PATH`. The scripts use only the standard library.
- Internet access to huggingface.co, and disk space for the models: 102 GiB for all of them. The size filters below select a subset.
- Nothing to install and no admin rights needed. Everything, including the Visual C++ runtime, is in `bin\`, and the scripts put `bin\` first on `PATH` so that another OpenVINO installation can't interfere.

## Quick check (about 5 minutes)

From the package folder:

```
python scripts\fetch_models.py --only Llama-3.2-1B    (0.75 GiB)
bin\hello_query_device.exe
```

`hello_query_device` should list `CPU`, `GGML` and, on a machine with an Intel GPU, `GPU`. Then run one model four ways. These lines are for Command Prompt; in PowerShell, write `$M = "..."` and use `$M` in place of `%M%`:

```
set M=models\bartowski__Llama-3.2-1B-Instruct-GGUF\Llama-3.2-1B-Instruct-Q4_K_M.gguf
set A=artifacts\Llama-3.2-1B-Instruct-Q4_K_M\d1.json
bin\ggml_llm.exe %M% %A% "The capital of France is" 32 GGML_BACKEND=Vulkan0
bin\ggml_llm.exe %M% %A% "The capital of France is" 32 GGML_BACKEND=CPU
bin\ggml_llm.exe %M% CPU "The capital of France is" 32
bin\ggml_llm.exe %M% GPU "The capital of France is" 32
```

The four lines are GGML GPU, GGML CPU, OV CPU and OV GPU, in that order.

Each run should print "The capital of France is Paris." and a timing line. The GPU run also prints a line like `ggml_vulkan: 0 = Intel(R) ... | uma: 1 | fp16: 1 | ...`, which shows the GPU ggml uses. Timings from these single runs include one-time GPU shader compilation; the full test warms up first.

## Full test

```
python scripts\run_tests.py --download --max-gib 2.5
python scripts\run_tests.py --download
python scripts\run_tests.py --only Llama-3.2,Qwen3 --download
```

The first command tests the 17 models up to 2.5 GiB (25 GiB of downloads). The second tests all 34 models (102 GiB, several hours). The third tests only models whose names contain one of the given substrings; here that's 10 models, 21 GiB.

Options:

| Option | Effect |
| :-- | :-- |
| `--download` | fetch the selected models first. Without it, only models already downloaded are tested. |
| `--models-dir DIR` | use models stored elsewhere (default `models\`) |
| `--max-gib N`, `--only a,b` | select models by file size or by name |
| `--gpu NAME` | ggml's name for the GPU (default `Vulkan0`). With several GPUs, a run's `ggml_vulkan:` lines list them. |
| `--ov-gpu NAME` | OpenVINO device for the OV GPU baseline. The default is the first Intel GPU that `hello_query_device` lists; `none` skips the baseline. |
| `--tokens N` | maximum new tokens per run (default 32) |
| `--no-cpu`, `--no-gpu` | skip a pass |
| `--out DIR` | output folder (default `results\<computer>_<date>`) |

For each model, the test runs a CPU pass and a GPU pass:

- **CPU pass:** GGML CPU with the 1-token graph (`d1`), GGML CPU with the 8-token prefill bucket (`d1+b8`), and OV CPU.
- **GPU pass:** GGML GPU with `d1` and with `d1+b8`, plus OV GPU when OpenVINO lists an Intel GPU.

Every run uses greedy decoding on the prompt "The capital of France is", wrapped in the model's chat template. Every run also starts with one untimed generation, so the timings exclude one-time setup such as GPU shader compilation.

Results go to `results\<computer>_<date>\`:

| File | Contents |
| :-- | :-- |
| `summary.md` | one row per model |
| `system.txt` | CPU, RAM, GPU and driver, Vulkan device, OpenVINO devices |
| `cpu\e2e.md`, `gpu\e2e.md` | each run's text, load time, prompt and generated token counts |
| `*.log` | full output of each run |

To share the results, zip the results folder.

## Reading summary.md

`summary.md` has three tables: decode time per token (TPOT), time to the first token (TTFT), and whether the output text matches.

- The ratio columns divide an OpenVINO TPOT by the GGML GPU TPOT. Above 1 means the GGML GPU path decodes faster. "GGML GPU vs OV GPU" is the comparison of interest on an Intel GPU: llama.cpp's kernels against OpenVINO's own GPU plugin on the same hardware.
- **GGML GPU = GGML CPU** compares the same graph on two backends. It should usually say yes. A late difference is normal: slightly different float results can flip a close greedy choice.
- **GGML CPU = OV CPU** often says no after a few tokens. OpenVINO's CPU plugin uses a u8 KV cache and dynamic int8 activations by default; ggml keeps the KV cache in f16. OpenVINO's GPU plugin has its own precision defaults, so **GGML GPU = OV GPU** drifts the same way.

Expected gaps, which are not problems with the package:

- **Qwen3.5 (4 models):** no `b8` results. Their recurrent layers cannot take padded batches, so the plugin rejects prefill buckets by design.
- **OV CPU and OV GPU missing for 18 models:** GenAI's GGUF path doesn't support their architecture (gemma3, granite, exaone, hunyuan-dense) or their tensor types (gemma4, Phi, Bonsai, Qwen3.5). This happens before any device is used, so these models run only on the GGML device.
- **No OV GPU columns** on a machine without an Intel GPU. OpenVINO's GPU plugin supports Intel GPUs only. It also lists an NVIDIA GPU, through NVIDIA's OpenCL, but inference fails there, so the scripts don't use it.
- **"Out of memory":** the model didn't fit the GPU. An integrated GPU shares system RAM, so this depends on how much RAM is installed.
- **gemma-4-E2B and E4B fail on the GPU** of the Arc 140V (Lunar Lake), at load and without an error message. The cause isn't confirmed yet; see the [test report](reports/lunarlake_core-ultra-7-268v/TEST_REPORT.md).
- **Answers that run on past the end** (for example "Paris" followed by "thought…"): Gemma 4 and the Phi models end a turn with a token that isn't their `eos_token_id`, and GenAI's GGUF path stops only at `eos_token_id`. The driver's `STOP_IDS` option fixes this for manual runs:
  - Gemma 4: `<turn|>` = 106
  - Phi-3 and Phi-3.5: `<|end|>` = 32007
  - Phi-4-mini: `<|end|>` = 200020

  To find a token's id, run `python scripts\gguf_meta.py <model.gguf> --find "<|end|>"`.

## The driver: bin\ggml_llm.exe

```
ggml_llm <model.gguf> <artifact.json | device> "<prompt>" [max_new_tokens] [KEY=value ...]
```

- With an artifact, the driver compiles it on the `GGML` device: `compile_model(artifact, "GGML", {GGML_WEIGHTS: model.gguf, ...})`. It then generates through `LLMPipeline`, using a tokenizer built from the GGUF.
- With a device name (`CPU`, `GPU`) instead, it uses GenAI's own GGUF path on that device.
- It prints the generated text, then the load time, prompt and generated token counts, TTFT, TPOT and tokens/s.

`KEY=value` options:

| Option | Effect |
| :-- | :-- |
| `GGML_BACKEND=Vulkan0` / `CPU` | ggml backend. The default picks a discrete GPU, then an integrated GPU, then the CPU. |
| `GGML_BUCKETS=8:<d8.json>` | batched prefill with the 8-token graph |
| `GGML_PAGED=YES` | continuous batching (`LLMPipeline` with a `SchedulerConfig`). Prefill buckets for this mode must be dumped with `ALL_LOGITS=1`; the included `d8.json` files are not. |
| `STOP_IDS=106,1` | extra stop tokens for GenAI |
| `WARMUP=1` | one untimed generation before the measured one |

The artifacts are shape-static. Each is valid only for its token count and a 512-entry KV cache, so prompt plus answer must stay under 512 tokens. Other context sizes or models need new dumps from ggml-cgraph-generator (not included; see `VERSIONS.txt`).

## Not included

OpenVINO's NPU plugin.

## Troubleshooting

- **No Vulkan device, or a `ggml_vulkan` error:** update the Intel graphics driver and check `vulkaninfo --summary`.
- **A DLL fails to load:** run the tools from this package's `bin\`, as the scripts do, rather than from a shell where another OpenVINO's `bin` comes first on `PATH`.
- **A download stopped:** run `fetch_models.py` again. It resumes partial files and skips complete ones.
