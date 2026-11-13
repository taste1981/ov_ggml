# Building the GGML device stack from source (Windows)

These steps rebuild everything in this package from source: OpenVINO with the GGML plugin, OpenVINO GenAI, the test driver, OpenVINO's and GenAI's samples, and the tools that dump the graph artifacts. All of it builds on Windows with Visual Studio; WSL is not needed.

Tested configuration: Windows Server 2025, Visual Studio 2022 17.14 (MSVC 19.44), CMake 4.3.2, Git 2.47, Vulkan SDK 1.4.363, 2× 32-core Xeon (128 threads).

## How the pieces fit

```
llama.cpp (CPU-only) ──► ggml-cgraph-generator ──► artifacts\<model>\d1.json, d8.json   offline, once per model
                          dump_cgraph, dump_vision

OpenVINO ggml_plugin ──► OpenVINO GenAI ggml_adapter ──► ggml_llm (this package's driver)
 GGML plugin + ggml                                   └─► GenAI samples (greedy_causal_lm, ...)
 CPU / Vulkan backends, CPU and GPU plugins, samples
```

At run time, OpenVINO's `GGML` device loads an artifact (the model's graph) and reads the weights from the original `.gguf`. llama.cpp is needed only to create artifacts.

## 1. Prerequisites

| | |
| :-- | :-- |
| Visual Studio 2022 | the "Desktop development with C++" workload (MSVC v143, Windows SDK) |
| CMake 3.23 or later | CMake 4.x also needs `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`, because some third-party projects declare old minimums |
| Git for Windows | |
| Vulkan SDK | only for ggml's Vulkan backend (glslc, headers, `vulkan-1.lib`) |
| Python 3 | only for the package's test scripts; the builds don't need it |

Installing the Vulkan SDK:

- **With admin rights:** `winget install KhronosGroup.VulkanSDK`, then open a new terminal so that `VULKAN_SDK` is set.
- **Without admin rights:** download the installer from https://sdk.lunarg.com and install copy-only into any folder. Copy-only sets no environment variables, so set them in the terminal you build in:

  ```bat
  vulkansdk-windows-X64-1.4.363.0.exe --root C:\VulkanSDK\1.4.363.0 --accept-licenses --default-answer --confirm-command install copy_only=1
  set VULKAN_SDK=C:\VulkanSDK\1.4.363.0
  set PATH=%VULKAN_SDK%\Bin;%PATH%
  ```

Disk space and build time on the test machine:

| | Source | Build | Time |
| :-- | --: | --: | :-- |
| OpenVINO | 2.3 GB | 14 GB | about 30 min (CPU, GGML, Vulkan, GPU plugin) |
| GenAI | 0.1 GB | 1.6 GB | about 12 min |
| llama.cpp, generator | 0.5 GB | (included) | a few minutes |

A laptop takes several times longer. Use a short root folder such as `C:\ov`, because MSBuild paths get long. The commands below are for Command Prompt (cmd.exe) and assume the root `C:\ov`. They use `-j 16`, which runs that many MSBuild projects in parallel; each project also compiles files in parallel, so 16 is enough even on large machines.

## 2. Get the sources

Several Windows fixes are in pull requests to Yamini's repositories that may not be merged yet. Until they are, start from the `taste1981` branches, which contain them:

| Repository | Branch to use now | Contains | Once the PRs are merged |
| :-- | :-- | :-- | :-- |
| OpenVINO | `taste1981/openvino` `ggml_plugin_windows`, plus a merge of `ggml_plugin_mixed_batch` | `ynimmaga/openvino` `ggml_plugin` + #101 + #102 | `ynimmaga/openvino` `ggml_plugin` |
| GenAI | `taste1981/openvino.genai` `ggml_adapter_windows` | `ynimmaga/openvino.genai` `ggml_adapter` + #1 | `ynimmaga/openvino.genai` `ggml_adapter` |
| generator | `taste1981/ggml-cgraph-generator` `cmake-cross-platform`, plus a merge of `dump_scale_rows_constant` | `plugin_artifacts` + #1 + the inp_scale_rows fix | `ynimmaga/ggml-cgraph-generator` `plugin_artifacts` |

Start OpenVINO from `ggml_plugin_windows`, not from Yamini's `ggml_plugin` with the fix pulled in afterwards. That branch makes git write the `.patch` files with LF line endings, and a checkout made before that rule was in place keeps CRLF; see the Windows notes at the end.

```bat
mkdir C:\ov & cd /d C:\ov

git clone --filter=blob:none -b ggml_plugin_windows https://github.com/taste1981/openvino.git
cd openvino
git pull --no-rebase --no-edit https://github.com/taste1981/openvino.git ggml_plugin_mixed_batch
git submodule update --init --depth 1 src/plugins/auto/thirdparty/ipf src/plugins/intel_cpu/thirdparty/mlas ^
    src/plugins/intel_cpu/thirdparty/onednn src/plugins/intel_gpu/thirdparty/onednn_gpu thirdparty/gflags/gflags ^
    thirdparty/ittapi/ittapi thirdparty/json/nlohmann_json thirdparty/level_zero/level-zero thirdparty/ocl/cl_headers ^
    thirdparty/ocl/clhpp_headers thirdparty/ocl/icd_loader thirdparty/pugixml thirdparty/telemetry thirdparty/xbyak ^
    thirdparty/zlib/zlib
cd ..

git clone -b ggml_adapter_windows https://github.com/taste1981/openvino.genai.git
git -C openvino.genai submodule update --init

git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout dd6b945be

git clone -b cmake-cross-platform https://github.com/taste1981/ggml-cgraph-generator.git
git -C ggml-cgraph-generator pull --no-rebase --no-edit https://github.com/taste1981/ggml-cgraph-generator.git dump_scale_rows_constant
```

The `git pull` lines create a merge commit. If git asks who you are, set `git config --global user.name` and `user.email`, or add `-c user.name=me -c user.email=me@example.com` after `git`.

The submodule list is what this configuration needs: CPU, GPU, GGML and AUTO plugins, IR and GGUF frontends, samples. `git submodule update --init --recursive` also works but downloads much more.

## 3. Build OpenVINO with the GGML plugin

Run this in a terminal where the Vulkan SDK is set up (see step 1):

```bat
cmake -S openvino -B build\openvino -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF ^
    -DENABLE_GGML_CGRAPH_LOADER=ON -DENABLE_GGML_VULKAN=ON ^
    -DENABLE_INTEL_GPU=ON -DENABLE_INTEL_NPU=OFF ^
    -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_TF_FRONTEND=OFF ^
    -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF ^
    -DENABLE_PYTHON=OFF -DENABLE_JS=OFF -DENABLE_TESTS=OFF -DENABLE_SAMPLES=ON ^
    -DENABLE_AUTO=OFF -DENABLE_AUTO_BATCH=OFF -DENABLE_TEMPLATE=OFF -DENABLE_PROXY=OFF
cmake --build build\openvino --config Release -j 16
```

| Option | Effect |
| :-- | :-- |
| `ENABLE_GGML_CGRAPH_LOADER=ON` | builds the cgraph loader and the `GGML` device plugin, and fetches ggml v0.23.0 with two local patches |
| `ENABLE_GGML_VULKAN=ON` | builds ggml's Vulkan backend and needs `glslc`. With `OFF`, only ggml's CPU backend is built and no Vulkan SDK is needed. |
| `ENABLE_INTEL_GPU=ON` | OpenVINO's own GPU plugin, the baseline on Intel GPUs. With `OFF`, the build is faster and the `onednn_gpu`, `ocl` and `level-zero` submodules aren't needed. |
| frontend, Python and test options | turned off for a smaller build. The GGML path needs none of them; GenAI's own GGUF path uses the GGUF frontend, which stays on. |

The binaries go to `openvino\bin\intel64\Release\` inside the source tree, not into the build folder: `openvino.dll`, the plugins, `openvino_ggml_cgraph_loader.dll`, `ggml*.dll`, `OpenCL.dll`, TBB, and the samples. Check the result with `openvino\bin\intel64\Release\hello_query_device.exe`, which should list `CPU`, `GGML` and, on a machine with an Intel GPU, `GPU`.

## 4. Build OpenVINO GenAI

GenAI builds directly against the OpenVINO build folder; no install step is needed.

```bat
cmake -S openvino.genai -B build\genai -G "Visual Studio 17 2022" -A x64 ^
    -DOpenVINO_DIR=C:\ov\build\openvino ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF ^
    -DENABLE_PYTHON=OFF -DENABLE_JS=OFF -DENABLE_TESTS=OFF -DENABLE_TOOLS=OFF -DENABLE_SAMPLES=OFF ^
    -DENABLE_XGRAMMAR=OFF -DENABLE_MISAKI_CPP=OFF -DENABLE_GGUF=ON -DBUILD_TOKENIZERS=ON
cmake --build build\genai --config Release -j 16
```

Output: `build\genai\openvino_genai\openvino_genai.dll` and `openvino_tokenizers.dll`.

`ENABLE_SAMPLES` is off on purpose. With it on, four of GenAI's sample folders download and build OpenCV from source, and the GGML device doesn't need OpenCV. Step 5 builds the text-generation samples on their own instead.

## 5. Build the samples

All three builds below find OpenVINO and GenAI through their build folders.

**OpenVINO samples** (`hello_query_device`, `benchmark_app`, ...) are built in step 3 by `-DENABLE_SAMPLES=ON`.

**ggml_llm**, this package's test driver, is the program that uses the `GGML` device through GenAI. Its source is in `src\ggml_llm` of this package:

```bat
cmake -S <package>\src\ggml_llm -B build\ggml_llm -G "Visual Studio 17 2022" -A x64 ^
    -DOpenVINO_DIR=C:\ov\build\openvino -DOpenVINOGenAI_DIR=C:\ov\build\genai
cmake --build build\ggml_llm --config Release
```

**GenAI text-generation samples** (`greedy_causal_lm`, `chat_sample`, `benchmark_genai`, ...): GenAI's `samples\cpp\text_generation` folder is not a standalone CMake project, so `src\genai_samples` in this package wraps it:

```bat
cmake -S <package>\src\genai_samples -B build\genai_samples -G "Visual Studio 17 2022" -A x64 ^
    -DGENAI_SOURCE_DIR=C:\ov\openvino.genai -DOpenVINO_DIR=C:\ov\build\openvino -DOpenVINOGenAI_DIR=C:\ov\build\genai
cmake --build build\genai_samples --config Release -j 16
```

The executables are in `build\genai_samples\text_generation\Release\`. These samples take a model path and an OpenVINO device, so they run GenAI's own GGUF path; for example, `greedy_causal_lm.exe model.gguf "The capital of France is"` uses the CPU. They can't drive the `GGML` device, which needs an artifact plus a `GGML_WEIGHTS` property. That's what `ggml_llm` is for.

To run anything from the build folders, put both binary folders on `PATH` first:

```bat
set PATH=C:\ov\openvino\bin\intel64\Release;C:\ov\build\genai\openvino_genai;%PATH%
```

## 6. Build the artifact dumper and dump new artifacts

The artifacts in this package cover 34 models with a 512-entry KV cache. Other models, context sizes or prefill buckets need new dumps.

Build llama.cpp CPU-only. If a GPU backend is present at dump time, dumps with `N_TOKENS` of 16 or more record backend-specific repacked weights that the loader can't use. Then build the generator against it:

```bat
cmake -S llama.cpp -B llama.cpp\build -G "Visual Studio 17 2022" -A x64 ^
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DGGML_VULKAN=OFF
cmake --build llama.cpp\build --config Release --target llama mtmd -j 16

cmake -S ggml-cgraph-generator -B ggml-cgraph-generator\build -G "Visual Studio 17 2022" -A x64 -DLLAMA_CPP_DIR=C:\ov\llama.cpp
cmake --build ggml-cgraph-generator\build --config Release -j 16
```

This produces `ggml-cgraph-generator\build\Release\dump_cgraph.exe` and `dump_vision.exe`, with the llama.cpp DLLs copied next to them. For `dump_whisper`, also clone whisper.cpp (tested at `d2c6f05a`), build its ggml libraries (`cmake --build whisper.cpp\build --config Release --target ggml`), and add `-DWHISPER_CPP_DIR=C:\ov\whisper.cpp` to the generator's configure.

Dumping, from any folder (the outputs are written to the current folder):

```bat
set D=C:\ov\ggml-cgraph-generator\build\Release
%D%\dump_cgraph.exe model.gguf d1.json 512
set N_TOKENS=8
%D%\dump_cgraph.exe model.gguf d8.json 512
set ALL_LOGITS=1
%D%\dump_cgraph.exe model.gguf d8_all.json 512
set N_TOKENS=& set ALL_LOGITS=
```

- `d1.json` is the single-token decoder, which every run needs. 512 is the KV cache size the artifact is fixed to.
- `d8.json` is an 8-token prefill bucket (`GGML_BUCKETS=8:d8.json`).
- `d8_all.json` is the same bucket with every row's logits, which the paged mode (`GGML_PAGED=YES`) requires.
- Vision and whisper dumps (`dump_vision`, `DUMP_VLM_DECODER=1`, `dump_whisper`) are described in the generator's README.

**Checking a new artifact against llama.cpp:** dump a reference with `set REF_OUT=ref.bin` and `set NO_FILL=1`, then run it through `ggml_llm` with `set GGML_REF=ref.bin`. The loader compares every node of the first decode step with llama.cpp's values and prints the first nodes that diverge, plus the final output's cosine similarity. Expect about 0.9999. Before the inp_scale_rows fix, Gemma 3 showed 0.0 here.

## 7. Assembling a package like this one

Put these files in one folder, as this package's `bin\` does:

- every DLL from `openvino\bin\intel64\Release`, except the `openvino_template_extension.dll` sample, plus `hello_query_device.exe`;
- `openvino_genai.dll` and `openvino_tokenizers.dll` from `build\genai\openvino_genai`;
- `ggml_llm.exe`;
- the Visual C++ runtime DLLs (`msvcp140*.dll`, `vcruntime140*.dll`) from `C:\Program Files\Microsoft Visual Studio\2022\<edition>\VC\Redist\MSVC\<version>\x64\Microsoft.VC143.CRT`.

Windows loads DLLs from the executable's folder first, and OpenVINO finds its plugins next to `openvino.dll`. The only DLL that must come from the system is `vulkan-1.dll`, which ships with the GPU driver.

## Windows notes

- **CRLF patch files:** OpenVINO and GenAI patch their dependencies at configure time with `git apply`. Their `.gitattributes` (`* text=auto`) gives Windows checkouts CRLF `.patch` files, which `git apply` rejects against LF sources when `core.autocrlf` is false. The fixed branches add `*.patch text eol=lf`. For a checkout made without that rule, delete the `.patch` files and check them out again (`git checkout -- <folder>`).
- **OpenMP:** ggml is built without OpenMP on Windows. With MSVC's OpenMP, unloading the plugin crashed the process at exit.
- **Several Visual Studio installations:** `-G "Visual Studio 17 2022"` makes CMake pick a VS 2022 instance. To choose one, add `-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Enterprise"`.
- **ICU:** the tokenizers build ICU only with `-DREGENERATE_PRECOMPILED_CHARSMAP=ON`. That build calls `msbuild` from `PATH`, so it needs a Developer Command Prompt. The default doesn't build ICU.
- **Re-running configure** with a changed option (for example, adding `-DENABLE_INTEL_GPU=ON` later) keeps the other cached options. Only the affected targets rebuild.
