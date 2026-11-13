"""Run models end to end through ggml_llm: GGML device (single-token artifact, then with an 8-token
prefill bucket) and GenAI's native GGUF path on CPU; tabulate timings and whether the text matches.

usage: python -I e2e_llm.py <ggml_llm.exe> <validated_dir> <sweep_dir> <out_dir> [max_new_tokens]
           [--backend NAME] [--native CPU,GPU] [--max-gib N] [--only a,b] [--ref <e2e.json>]...
           [--ref-mode MODE] [--warmup]
--backend sets GGML_BACKEND (e.g. Vulkan0, CPU). --native lists the OpenVINO devices that also run
the model through GenAI's own GGUF path (default CPU). Text is compared with the CPU run, or with
MODE ("GGML d1" by default) of the earlier runs given by --ref (later files win).
The OpenVINO, GenAI and ggml DLLs must sit next to ggml_llm.exe or be on PATH.
Writes <out_dir>/e2e.json and <out_dir>/e2e.md.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROMPT = "The capital of France is"
METRICS = re.compile(r"\n\nload ([\d.]+) ms, prompt (\d+) tokens, generated (\d+) tokens, "
                     r"TTFT ([\d.naif-]+) ms, TPOT ([\d.naif-]+) ms")


def run(cmd, log):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1800)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    res = {"exit": r.returncode}
    m = METRICS.search(r.stdout)
    if m:
        res["text"] = r.stdout[:m.start()]
        res.update(load_s=round(float(m[1]) / 1000, 1), prompt=int(m[2]), generated=int(m[3]),
                   ttft_ms=round(float(m[4]), 1), tpot_ms=round(float(m[5]), 1))
    else:
        res["error"] = [l for l in (r.stderr or r.stdout).splitlines() if l.strip()][-2:]
    return res


def load_refs(paths, mode):
    refs = {}
    for path in paths:
        for e in json.loads(Path(path).read_text(encoding="utf-8")):
            text = e["runs"].get(mode, {}).get("text")
            if text is not None:
                refs[e["model"]] = text
    return refs


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("exe"), ap.add_argument("vdir"), ap.add_argument("sweep"), ap.add_argument("out")
    ap.add_argument("n", nargs="?", default="32")
    ap.add_argument("--backend")
    ap.add_argument("--native", default="CPU",
                    help="OpenVINO devices for GenAI's own GGUF path, comma-separated ('' for none)")
    ap.add_argument("--skip-cpu", action="store_true", help="same as --native ''")
    ap.add_argument("--max-gib", type=float)
    ap.add_argument("--only", help="comma-separated substrings; run only models whose name contains one")
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--ref-mode", default="GGML d1")
    ap.add_argument("--warmup", action="store_true", help="untimed generate before each measured one")
    a = ap.parse_args()
    exe, vdir, sweep, out = (Path(p).resolve() for p in (a.exe, a.vdir, a.sweep, a.out))
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    ggufs = {Path(i["path"]).stem: (vdir / i["repo"].replace("/", "__") / Path(i["path"]).name, i["size"])
             for i in manifest}
    refs = load_refs(a.ref, a.ref_mode)
    natives = [] if a.skip_cpu else [d for d in a.native.split(",") if d]
    warm = ["WARMUP=1"] if a.warmup else []
    extra = ([f"GGML_BACKEND={a.backend}"] if a.backend else []) + warm
    tag = f"GGML[{a.backend}]" if a.backend else "GGML"
    results = []
    for mdir in sorted(p for p in sweep.iterdir() if (p / "d1.json").exists()):
        gguf, size = ggufs.get(mdir.name, (None, 0))
        if not gguf or not gguf.exists() or (a.max_gib and size / 2**30 > a.max_gib):
            continue
        if a.only and not any(s.lower() in mdir.name.lower() for s in a.only.split(",")):
            continue
        modes = {f"{tag} d1": [exe, gguf, mdir / "d1.json", PROMPT, a.n] + extra}
        if (mdir / "d8.json").exists():
            modes[f"{tag} d1+b8"] = modes[f"{tag} d1"] + [f"GGML_BUCKETS=8:{mdir / 'd8.json'}"]
        for dev in natives:
            modes[dev] = [exe, gguf, dev, PROMPT, a.n] + warm
        entry = {"model": mdir.name, "runs": {}}
        for mode, cmd in modes.items():
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", mode)
            entry["runs"][mode] = run(cmd, out / f"{mdir.name}.{safe}.log")
            print(json.dumps({"model": mdir.name, "mode": mode,
                              **{k: v for k, v in entry["runs"][mode].items() if k != "text"}}), flush=True)
        results.append(entry)
    (out / "e2e.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    against = "ref" if a.ref else "CPU"
    rows = [f"| Model | Mode | Exit | Load s | Prompt | Gen | TTFT ms | TPOT ms | Same text as {against} | Text |",
            "| :-- | :-- | :-: | --: | --: | --: | --: | --: | :-: | :-- |"]
    for e in results:
        ref = refs.get(e["model"]) if a.ref else e["runs"].get("CPU", {}).get("text")
        for mode, r in e["runs"].items():
            text = r.get("text", " / ".join(r.get("error", [])))
            same = "" if mode in natives or ref is None or "text" not in r else ("yes" if r["text"] == ref else "**no**")
            brief = text.replace("\n", " ").replace("|", "\\|").strip()[:70]
            rows.append(f"| {e['model']} | {mode} | {r['exit']} | {r.get('load_s', '')} | {r.get('prompt', '')} | "
                        f"{r.get('generated', '')} | {r.get('ttft_ms', '')} | {r.get('tpot_ms', '')} | {same} | {brief} |")
    (out / "e2e.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
