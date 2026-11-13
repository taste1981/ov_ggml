"""Merge the cpu and gpu passes of run_tests.py into <out>/summary.md.

usage: python summarize.py <out_dir> [--models-dir DIR]
Three tables: decode time per token (TPOT), time to the first token (TTFT), and text agreement, for the
GGML device on ggml's CPU and GPU backends and for GenAI's own GGUF path on the OpenVINO CPU plugin and,
when it ran, the OpenVINO GPU plugin.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWN = [("recurrent models take 1-token graphs only", "no prefill buckets for recurrent models (by design)"),
         ("Unsupported model architecture", "GenAI GGUF: architecture not supported"),
         ("gguf_tensor_to_f16 failed", "GenAI GGUF: tensor type not supported"),
         ("not enough memory", "out of memory"),
         ("ErrorOutOfDeviceMemory", "out of device memory")]


def load(path):
    return {e["model"]: e["runs"] for e in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}


def ggml(runs, suffix):
    """The GGML run whose mode ends with suffix (e.g. '] d1'), whatever the backend."""
    return next((r for m, r in runs.items() if m.startswith("GGML[") and m.endswith(suffix)), {})


def native(runs):
    """The GenAI-native run of a pass (mode = an OpenVINO device name)."""
    return next((r for m, r in runs.items() if not m.startswith("GGML[")), {})


CRASHES = {3221225477: "access violation", 3221226505: "abort (stack buffer overrun)", 3: "abort"}


def why(run):
    err = " ".join(run.get("error", []))
    known = next((short for key, short in KNOWN if key in err), None)
    if known:
        return known
    code = run.get("exit")
    kind = CRASHES.get(code, f"exit code {code}")
    # ggml_vulkan device lines contain '|', which would split the table cell
    last = err[:60].replace("|", "/") if err and "ggml_vulkan:" not in err else "no error message"
    return f"failed: {kind}, {last}"


def ms(run, key):
    v = run.get(key)
    return "" if v is None else f"{v:g}"


def ratio(num, den):
    return f"{num['tpot_ms'] / den['tpot_ms']:.1f}x" if num.get("tpot_ms") and den.get("tpot_ms") else ""


def same(x, y):
    return "" if "text" not in x or "text" not in y else ("yes" if x["text"] == y["text"] else "no")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    cpu, gpu = load(out / "cpu" / "e2e.json"), load(out / "gpu" / "e2e.json")
    manifest_path = Path(a.models_dir) / "manifest.json"
    if not manifest_path.exists():
        manifest_path = ROOT / "models" / "manifest.json"
    sizes = {Path(i["path"]).stem: i["size"] / 2**30 for i in json.loads(manifest_path.read_text(encoding="utf-8"))}
    has_ovg = any(native(r) for r in gpu.values())

    rows = []
    for model in sorted(set(cpu) | set(gpu), key=lambda m: (sizes.get(m, 0), m)):
        c, g = cpu.get(model, {}), gpu.get(model, {})
        rows.append((model, sizes.get(model, 0), ggml(c, "] d1"), ggml(c, "] d1+b8"), native(c),
                     ggml(g, "] d1"), ggml(g, "] d1+b8"), native(g)))

    lines = ["# GGML device vs OpenVINO: summary", ""]
    sysinfo = out / "system.txt"
    if sysinfo.exists():
        lines += ["```", sysinfo.read_text(encoding="utf-8", errors="replace").strip(), "```", ""]
    lines += ["GGML = OpenVINO's GGML device replaying llama.cpp's graph (CPU or Vulkan backend). "
              "OV = GenAI's own GGUF path on an OpenVINO plugin. All runs are warm, greedy, 32 new tokens.", ""]

    head = "| Model | GiB | GGML CPU | GGML GPU | OV CPU |" + (" OV GPU |" if has_ovg else "") + \
           " GGML GPU vs OV CPU |" + (" GGML GPU vs OV GPU |" if has_ovg else "") + " Notes |"
    lines += ["## Decode: time per token (TPOT, ms)", "", head,
              "| :-- | --: | --: | --: | --: |" + (" --: |" if has_ovg else "") + " --: |" + (" --: |" if has_ovg else "") + " :-- |"]
    for model, gib, gc, gc8, oc, gg, gg8, og in rows:
        notes = "; ".join(sorted({why(r) for r in (gc, gc8, oc, gg, gg8, og) if r and r.get("exit")}))
        lines.append(f"| {model} | {gib:.2f} | {ms(gc, 'tpot_ms')} | {ms(gg, 'tpot_ms')} | {ms(oc, 'tpot_ms')} |" +
                     (f" {ms(og, 'tpot_ms')} |" if has_ovg else "") + f" {ratio(oc, gg)} |" +
                     (f" {ratio(og, gg)} |" if has_ovg else "") + f" {notes} |")

    lines += ["", "Ratios above 1 mean the GGML GPU path decodes faster.", "",
              "## First token (TTFT, ms)", "", "d1 prefills one token per step; b8 uses the 8-token bucket.", "",
              "| Model | Prompt tokens | GGML CPU d1 | GGML CPU b8 | GGML GPU d1 | GGML GPU b8 | OV CPU |" +
              (" OV GPU |" if has_ovg else ""),
              "| :-- | --: | --: | --: | --: | --: | --: |" + (" --: |" if has_ovg else "")]
    for model, gib, gc, gc8, oc, gg, gg8, og in rows:
        prompt = next((r["prompt"] for r in (gc, gg, oc, og) if r.get("prompt")), "")
        lines.append(f"| {model} | {prompt} | {ms(gc, 'ttft_ms')} | {ms(gc8, 'ttft_ms')} | {ms(gg, 'ttft_ms')} | "
                     f"{ms(gg8, 'ttft_ms')} | {ms(oc, 'ttft_ms')} |" + (f" {ms(og, 'ttft_ms')} |" if has_ovg else ""))

    lines += ["", "## Same output text", "",
              "GGML GPU vs GGML CPU compares one graph on two backends; late differences are float rounding. "
              "OV CPU uses a u8 KV cache and int8 activations by default, so it often drifts from GGML after a few tokens.",
              "", "| Model | GGML GPU = GGML CPU | GGML CPU = OV CPU |" + (" GGML GPU = OV GPU |" if has_ovg else ""),
              "| :-- | :-: | :-: |" + (" :-: |" if has_ovg else "")]
    for model, gib, gc, gc8, oc, gg, gg8, og in rows:
        lines.append(f"| {model} | {same(gg, gc)} | {same(gc, oc)} |" + (f" {same(gg, og)} |" if has_ovg else ""))

    lines += ["", "Per-run details (text, load time, token counts): cpu/e2e.md and gpu/e2e.md; "
              "each run's full output is in the .log files next to them."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
