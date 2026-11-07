import os
import argparse
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
OUT = ROOT / "study/channel/results/tables/machine_environment.json"
MANIFEST = ROOT / "study/channel/results/raw/machine_environment.manifest.json"
FROZEN = ROOT / "study/_shared/env_versions.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_proc(fpath, key):
    for ln in Path(fpath).read_text().splitlines():
        if ln.startswith(key):
            return ln.split(":", 1)[1].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timpa", action="store_true",
                    help="replace the existing table with a reading of this machine")
    if OUT.exists() and not ap.parse_args().timpa:
        print("STOP: machine_environment.json exists and this script reads the machine it runs on.")
        print("The table records the workstation the measurements were taken on. Pass --timpa only if "
              "you mean to replace that record.")
        raise SystemExit(2)
    os_rel = {}
    for ln in Path("/etc/os-release").read_text().splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            os_rel[k] = v.strip('"')
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True).stdout.strip().split(", ")
    mem_kb = int(read_proc("/proc/meminfo", "MemTotal").split()[0])
    frozen = json.loads(FROZEN.read_text())

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "_apa_ini": ("Specification of the machine on which every number was measured, compared with the "
                     "frozen record env_versions.json."),
        "prosesor": read_proc("/proc/cpuinfo", "model name"),
        "thread_logis": int(subprocess.run(["nproc"], capture_output=True, text=True).stdout),
        "memori_terbaca_gib": round(mem_kb / 1024 / 1024, 2),
        "memori_terpasang_gb": 96,
        "catatan_memori": ("kernel MemTotal 93.87 GiB, the rest reserved by firmware and the integrated GPU; the "
                           "module type cannot be read without root, and the AM5 platform of this processor "
                           "supports DDR5 only"),
        "gpu": gpu[0], "vram_mib": int(gpu[1]), "driver": gpu[2],
        "os": os_rel.get("PRETTY_NAME"), "kernel_sekarang": platform.release(),
        "catatan_beku_9agu": {"gpu": frozen.get("gpu"), "platform": frozen.get("platform"),
                              "python": frozen.get("python")},
        "pembanding": {
            "driver_sama_dengan_catatan_beku": gpu[2] in str(frozen.get("gpu")),
            "kernel_catatan_beku": (re.search(r"Linux-([\d.]+-\d+-\w+)", str(frozen.get("platform"))) or [None, None])[1],
        },
        "batas": ("recorded after the measurements; it shows the same machine is still in place (GPU "
                  "driver identical to the frozen record), not the exact state at each run. The kernel has "
                  "received point updates since 9 Aug"),
        "label_source": "deterministic (read from /proc, /etc/os-release, nvidia-smi)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "machine_environment_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/8_audits/machine_environment.py",
        "output": {"path": "results/tables/machine_environment.json", "sha256": sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))
    print(json.dumps({k: output[k] for k in ("prosesor", "thread_logis", "memori_terbaca_gib",
                                            "gpu", "vram_mib", "driver", "os", "kernel_sekarang",
                                            "pembanding")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
