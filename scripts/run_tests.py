"""Run the GGML-device CPU and GPU test passes on this machine and summarize them.

usage: python run_tests.py [--models-dir DIR] [--download] [--max-gib N] [--only a,b] [--gpu Vulkan0]
                           [--ov-gpu GPU|none] [--tokens 32] [--out DIR] [--no-cpu] [--no-gpu]
Pass "cpu": GGML device on ggml's CPU backend (d1, d1+b8) and GenAI's native GGUF path on the OpenVINO
CPU plugin. Pass "gpu": GGML device on the Vulkan backend (d1, d1+b8), text compared with the cpu pass,
and GenAI's native GGUF path on the OpenVINO GPU plugin when OpenVINO lists an Intel GPU.
Results go to results/<computer>_<date>/ (summary.md, cpu/, gpu/, system.txt).
"""
import argparse
import datetime
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
SCRIPTS = ROOT / "scripts"


def capture(cmd, env=None, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"({cmd[0]}: {e})"


def system_info(env):
    ps = ("$p = @(Get-CimInstance Win32_Processor); $s = Get-CimInstance Win32_ComputerSystem; "
          "'CPU: ' + $p[0].Name.Trim() + ' x' + $p.Count + ' (' + ($p | Measure-Object NumberOfCores -Sum).Sum + "
          "' cores, ' + ($p | Measure-Object NumberOfLogicalProcessors -Sum).Sum + ' threads)'; "
          "'RAM: ' + [math]::Round($s.TotalPhysicalMemory / 1GB) + ' GB'; "
          "Get-CimInstance Win32_VideoController | ForEach-Object { 'GPU: ' + $_.Name + ', driver ' + $_.DriverVersion }")
    parts = [f"Host: {platform.node()}  OS: {platform.platform()}  Date: {datetime.datetime.now():%Y-%m-%d %H:%M}",
             capture(["powershell", "-NoProfile", "-Command", ps])]
    vulkaninfo = shutil.which("vulkaninfo") or str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkaninfo.exe")
    vk = capture([vulkaninfo, "--summary"])
    keep = [l.strip() for l in vk.splitlines() if any(k in l for k in ("Vulkan Instance Version", "deviceName", "deviceType",
                                                                       "driverName", "driverInfo", "apiVersion", "GPU"))]
    parts.append("vulkaninfo --summary:\n  " + ("\n  ".join(keep) if keep else vk[:500]))
    out = capture([str(BIN / "hello_query_device.exe")], env=env)
    # hello_query_device prints each device as an unindented all-caps line ("[ INFO ] GPU"), then its properties
    devices, current = {}, None
    for l in out.splitlines():
        body = l[9:] if l.startswith("[ INFO ] ") else ""
        if body and body[0] not in ("\t", " ") and ":" not in body and body.strip().isupper():
            current = body.strip()
            devices[current] = ""
        elif current and "FULL_DEVICE_NAME :" in body:
            devices[current] = body.split("FULL_DEVICE_NAME :", 1)[1].strip()
    parts.append("OpenVINO devices: " + (", ".join(f"{d} ({n})" for d, n in devices.items()) if devices else out[-300:]))
    return "\n".join(parts), devices


def run(cmd, env):
    print(">", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], env=env).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--download", action="store_true", help="fetch the selected models first (fetch_models.py)")
    ap.add_argument("--max-gib", type=float, help="only models up to this many GiB")
    ap.add_argument("--only", help="comma-separated substrings of model names, e.g. Llama-3.2,Qwen3")
    ap.add_argument("--gpu", default="Vulkan0", help="ggml backend name of the GPU (see the log's ggml_vulkan lines)")
    ap.add_argument("--ov-gpu", help="OpenVINO GPU device for the native baseline (default: the first Intel GPU "
                                     "OpenVINO lists; 'none' to skip)")
    ap.add_argument("--tokens", default="32", help="max new tokens per run")
    ap.add_argument("--out")
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()

    models = Path(a.models_dir).resolve()
    out = Path(a.out).resolve() if a.out else ROOT / "results" / f"{platform.node()}_{datetime.datetime.now():%Y%m%d-%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = str(BIN) + os.pathsep + env.get("PATH", "")  # the package's DLLs win over any other install
    py = sys.executable
    filters = (["--max-gib", str(a.max_gib)] if a.max_gib else []) + (["--only", a.only] if a.only else [])

    if a.download and run([py, SCRIPTS / "fetch_models.py", "--models-dir", models] + filters, env) != 0:
        print("some downloads failed; continuing with the models that are complete")
    info, devices = system_info(env)
    # OpenVINO's GPU plugin supports Intel GPUs only, though it may also list other OpenCL GPUs
    ov_gpu = a.ov_gpu or next((d for d, n in devices.items() if d.startswith("GPU") and "Intel" in n), "none")
    info += f"\nOpenVINO GPU baseline: {ov_gpu}"
    (out / "system.txt").write_text(info + "\n", encoding="utf-8")
    print(info)

    e2e = [py, SCRIPTS / "e2e_llm.py", BIN / "ggml_llm.exe", models, ROOT / "artifacts"]
    if not a.no_cpu:
        run(e2e + [out / "cpu", a.tokens, "--backend", "CPU", "--native", "CPU", "--warmup"] + filters, env)
    if not a.no_gpu:
        ref = ["--ref", out / "cpu" / "e2e.json", "--ref-mode", "GGML[CPU] d1"] if (out / "cpu" / "e2e.json").exists() else []
        native = "" if ov_gpu == "none" else ov_gpu
        run(e2e + [out / "gpu", a.tokens, "--backend", a.gpu, "--native", native, "--warmup"] + ref + filters, env)
    run([py, SCRIPTS / "summarize.py", out, "--models-dir", models], env)
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
