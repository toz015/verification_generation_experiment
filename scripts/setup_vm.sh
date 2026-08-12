#!/usr/bin/env bash
# Provision the A100 VM and verify everything a sweep depends on, up front.
#
# The point of this script is ordering: every check that could fail is done
# BEFORE any weights are downloaded or any model is served, so a missing token
# or a broken driver surfaces in setup rather than halfway through a run.
#
# Usage:  bash scripts/setup_vm.sh
# Needs:  HF_TOKEN exported, or a prior `hf auth login`.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_DIR/results/vm_manifest.json"
MODELS=("Qwen/Qwen3-8B" "meta-llama/Llama-3.1-8B-Instruct")

say() { printf '\n=== %s ===\n' "$1"; }
die() { printf '\nFAIL: %s\n' "$1" >&2; exit 1; }

# --- 1. GPU -----------------------------------------------------------------
say "GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Driver not installed."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || die "nvidia-smi present but cannot talk to the driver.
Usual cause is a kernel upgrade without a DKMS rebuild. Fix:
  sudo apt-get install -y linux-headers-\$(uname -r)
  sudo dkms autoinstall && sudo modprobe nvidia"

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$GPU_MEM_MIB" -ge 30000 ] || die "GPU has ${GPU_MEM_MIB}MiB; an 8B model in bf16 needs ~16GB plus KV cache."

# --- 2. Disk ----------------------------------------------------------------
# ~11GB for vLLM and torch, ~16GB per model in bf16, plus room for logs.
say "Disk"
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "free on /: ${AVAIL_GB}G"
[ "$AVAIL_GB" -ge 50 ] || die "Only ${AVAIL_GB}G free; need ~45G for vLLM plus both models.
Resize from your workstation:
  gcloud compute disks resize <disk> --zone <zone> --size 200GB
then on the VM: sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1"

# --- 3. Toolchain -----------------------------------------------------------
say "Toolchain"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

cd "$REPO_DIR"
uv venv --python 3.12 >/dev/null 2>&1 || true
# The [gpu] extra pulls vllm, which is CUDA-only and installed on the VM alone.
# --torch-backend=cu128 is required: the default wheel is built for CUDA 13 and
# needs driver >= 580, while this VM runs 535.
uv pip install -q -e ".[gpu]" --torch-backend=cu128 || die "install failed"

# torch.cuda.is_available() only checks that the driver initialises, and returns
# True even when kernels cannot launch. Run a real matmul instead.
uv run python -c "
import torch
a = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
assert torch.isfinite((a @ a).float().sum())
print('  bf16 matmul on', torch.cuda.get_device_name(0), '- OK')
" || die "torch cannot launch GPU kernels.
Usually a CUDA/driver mismatch. Check 'nvidia-smi' for the driver's CUDA
version and reinstall with a matching --torch-backend (cu121/cu124/cu128)."

# --- 4. HuggingFace access --------------------------------------------------
# Checked before any download. Llama-3.1-8B is gated=manual, so a valid token
# is not sufficient: the account must also be approved for that specific repo.
say "HuggingFace access"
uv run python - "${MODELS[@]}" <<'PY' || die "model access check failed"
import sys
from huggingface_hub import auth_check
from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError

bad = []
for repo in sys.argv[1:]:
    try:
        auth_check(repo)
        print(f"  ok      {repo}")
    except GatedRepoError:
        bad.append(repo)
        print(f"  GATED   {repo}  <- accept the license on its model page")
    except LocalTokenNotFoundError:
        bad.append(repo)
        print(f"  NOTOKEN {repo}  <- run `hf auth login` or export HF_TOKEN")
    except Exception as exc:
        bad.append(repo)
        print(f"  ERROR   {repo}: {exc}")

if bad:
    print("\nBlocked on:", ", ".join(bad))
    sys.exit(1)
PY

# --- 5. Prefetch weights ----------------------------------------------------
say "Prefetching weights"
for m in "${MODELS[@]}"; do
  echo "  $m"
  uv run hf download "$m" --quiet || die "download failed for $m"
done

# --- 6. Manifest ------------------------------------------------------------
# Driver, CUDA and vLLM versions sit outside the uv lockfile's control, so they
# are recorded here; a later discrepancy in results is then diagnosable.
say "Manifest"
mkdir -p "$(dirname "$MANIFEST")"
uv run python - "$MANIFEST" <<'PY'
import json, subprocess, sys, platform, datetime

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return None

manifest = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": platform.node(),
    "kernel": platform.release(),
    "python": platform.python_version(),
    "gpu": sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
    "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    "cuda_runtime": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
    "vllm": sh("uv run python -c 'import vllm; print(vllm.__version__)'"),
    "torch": sh("uv run python -c 'import torch; print(torch.__version__)'"),
    "git_commit": sh("git rev-parse HEAD"),
}
with open(sys.argv[1], "w") as fh:
    json.dump(manifest, fh, indent=2)
print(json.dumps(manifest, indent=2))
PY

say "Ready"
cat <<'EOF'
Run the sweep directly - no server, no ports. One model is loaded at a time
and the whole batch runs in a single call:

  uv run python -m vgx.scifact.run_sweep --model Qwen/Qwen3-8B
  uv run python -m vgx.scifact.run_sweep --model meta-llama/Llama-3.1-8B-Instruct

Use tmux if your connection is unreliable; the run is resumable either way,
since completed generations are skipped on restart.
EOF
