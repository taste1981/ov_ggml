# PR gatekeeping

This package was built and tested with the changes in these pull requests to Yamini's (ynimmaga) branches. Until they're merged, build from their head branches as [BUILD.md](BUILD.md) describes. For each repository, the tested tree is the base branch with all of its PRs below merged.

| PR | Branch | Change | Tested head | Status (2026-11-13) |
| :-- | :-- | :-- | :-- | :-- |
| [ynimmaga/openvino#101](https://github.com/ynimmaga/openvino/pull/101) | `taste1981:ggml_plugin_windows` → `ggml_plugin` | Windows: 64-bit GGUF offsets, `.patch` files kept LF, ggml built without OpenMP | `9db0d581` | open |
| [ynimmaga/openvino#102](https://github.com/ynimmaga/openvino/pull/102) | `taste1981:ggml_plugin_mixed_batch` → `ggml_plugin` | Artifacts from current llama.cpp: mixed-batch inputs, zeroed graph inputs, a clean out-of-memory error | `fd63d104` | open |
| [ynimmaga/openvino.genai#1](https://github.com/ynimmaga/openvino.genai/pull/1) | `taste1981:ggml_adapter_windows` → `ggml_adapter` | Windows: `.patch` files kept LF | `409aa805` | open |
| [ynimmaga/ggml-cgraph-generator#1](https://github.com/ynimmaga/ggml-cgraph-generator/pull/1) | `taste1981:cmake-cross-platform` → `plugin_artifacts` | Cross-platform CMake build, for Windows | `8a43f556` | open |
| [ynimmaga/ggml-cgraph-generator#2](https://github.com/ynimmaga/ggml-cgraph-generator/pull/2) | `taste1981:dump_scale_rows_constant` → `plugin_artifacts` | Gemma 3 and 4: `inp_scale_rows` written into the artifact as a constant | `8718c49a` | open |
| [ynimmaga/ggml-cgraph-generator#3](https://github.com/ynimmaga/ggml-cgraph-generator/pull/3) | `taste1981:dump_swa_mask_roles` → `plugin_artifacts` | Sliding-window attention: `mask_roles` per windowed mask leaf, derived from llama.cpp's filled mask values; REF_OUT `.inputs` keyed by tensor so both same-named masks are kept | `a7665b8` | open |
| [ynimmaga/openvino#103](https://github.com/ynimmaga/openvino/pull/103) | `taste1981:swa_mask_windows` → `ggml_plugin` | Loader: map `mask_roles` onto every alias of the leaf, refuse unequal-width masks (swa_full=false dumps); runners: windowed masks for SWA layers; `GGML_DUMP_MASK` debug hook | `65f4bd98` | open |
| [ynimmaga/openvino#104](https://github.com/ynimmaga/openvino/pull/104) | `taste1981:weights_shared` → `ggml_plugin` | Loader: `CacheMode` (weights shared, caches optional); plugin: one weight upload per (GGML_WEIGHTS, GGML_BACKEND) via a weak registry; whisper cross graph shares the decoder's upload | `bc7fed7c` | open |
| [ynimmaga/openvino#105](https://github.com/ynimmaga/openvino/compare/weights_shared...taste1981:request_state) | `taste1981:request_state` → `taste1981:weights_shared` | Each infer request runs its own runner over the shared upload: per-request KV caches, paged block pool and recurrent state; template runner kept for the signature only | `54e13b0f` | open (create PR) |
