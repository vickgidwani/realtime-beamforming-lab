#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/setup_gpu_vm.sh PROJECT ZONE INSTANCE [--create-on-subnet SUBNET]
       [--ssh-tag TAG] [--inference-source PATH] [--patched-mach]

Set up the lab on a running Ubuntu 24.04 GCP L4 VM. To create one, specify an
existing subnet and an SSH firewall tag (default: inference-lab-ssh).
The Exercise 7 exporter comes from the sibling inference-serving-lab checkout
by default. Use --inference-source to select another checkout.
Use --patched-mach to also build the explicit-stream Mach variant.
USAGE
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ $# -lt 3 ]]; then usage >&2; exit 2; fi
project=$1
zone=$2
instance=$3
shift 3
repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
inference_source="$(dirname "$repo_root")/inference-serving-lab"
subnet=
ssh_tag=inference-lab-ssh
patched_mach=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --create-on-subnet) subnet=${2:?Supply a subnet}; shift ;;
        --ssh-tag) ssh_tag=${2:?Supply an SSH firewall tag}; shift ;;
        --inference-source) inference_source=${2:?Supply an inference lab checkout}; shift ;;
        --patched-mach) patched_mach=true ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
for identifier in "$project" "$zone" "$instance" "$ssh_tag"; do
    if [[ ! $identifier =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
        printf 'Invalid GCP identifier: %s\n' "$identifier" >&2
        exit 2
    fi
done

exporter_dir="$inference_source/src/inference_lab"
if [[ ! -f $exporter_dir/__init__.py || ! -f $exporter_dir/model.py ]] \
    || ! grep -q -- '--embedding-output' "$exporter_dir/export_model.py"; then
    printf 'Expected the Exercise 7 embedding exporter in %s. Use --inference-source to set its checkout.\n' "$exporter_dir" >&2
    exit 2
fi
revision=$(git -C "$repo_root" rev-parse HEAD)
gcp_options=(--project="$project" --zone="$zone" --quiet)
ssh_options=("${gcp_options[@]}" --ssh-flag='-o ConnectTimeout=8')

if [[ -n $subnet ]]; then
    gcloud compute instances create "$instance" "${gcp_options[@]}" \
        --machine-type=g2-standard-4 --subnet="$subnet" --tags="$ssh_tag" \
        --maintenance-policy=TERMINATE --boot-disk-type=pd-balanced --boot-disk-size=50GB \
        --image-project=deeplearning-platform-release \
        --image=common-cu129-ubuntu-2404-nvidia-580-v20261111 \
        --no-service-account --no-scopes --labels=purpose=realtime-beamforming-lab \
        --format='table(name,zone.basename(),machineType.basename(),status)'
fi

for attempt in {1..12}; do
    if gcloud compute ssh "$instance" "${ssh_options[@]}" --command='true'; then break; fi
    if [[ $attempt -eq 12 ]]; then exit 1; fi
    sleep 3
done

remote_stage=$(gcloud compute ssh "$instance" "${ssh_options[@]}" \
    --command='mktemp -d "$HOME/.beamforming-setup.XXXXXXXX"')
[[ $remote_stage =~ ^/[-a-zA-Z0-9_./]+$ ]]
git -C "$repo_root" bundle create - HEAD | \
    gcloud compute ssh "$instance" "${ssh_options[@]}" --command="cat > '$remote_stage/source.bundle'"
setup_files=("$exporter_dir/__init__.py" "$exporter_dir/export_model.py" "$exporter_dir/model.py")
if $patched_mach; then
    setup_files+=("$repo_root"/patches/mach-v0.2.1-{explicit-stream-async,submission-nvtx,wrapper-nvtx,prepare-beamform}.patch)
fi
gcloud compute scp "${setup_files[@]}" \
    "$instance:$remote_stage/" "${gcp_options[@]}"

# All setup runs as the SSH user so the benchmark uses the same venv and data cache.
gcloud compute ssh "$instance" "${ssh_options[@]}" \
    --command="bash -se -- '$remote_stage' '$revision' '$patched_mach'" <<'REMOTE'
set -euo pipefail
stage=$1
revision=$2
patched_mach=$3
trap 'rm -f "$stage/source.bundle" "$stage/__init__.py" "$stage/export_model.py" "$stage/model.py" "$stage"/mach-v0.2.1-{explicit-stream-async,submission-nvtx,wrapper-nvtx,prepare-beamform}.patch; rmdir "$stage"' EXIT
lab_dir="$HOME/realtime-beamforming-lab"
exec 9>"$HOME/.realtime-beamforming-setup.lock"
flock 9
export PATH="/usr/local/cuda/bin:$PATH"
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fx 'NVIDIA L4'

sudo -n apt-get update -qq
sudo -n apt-get install -y -qq build-essential python3-dev python3-venv
if ! command -v nsys >/dev/null || ! command -v ncu >/dev/null || ! command -v nvcc >/dev/null; then
    if ! dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null | grep -q 'install ok installed'; then
        curl -fsSL --retry 3 -o "$stage/cuda-keyring.deb" \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        sudo -n dpkg -i "$stage/cuda-keyring.deb"
        unlink "$stage/cuda-keyring.deb"
    fi
    sudo -n apt-get update -qq
    sudo -n apt-get install -y -qq cuda-nvcc-12-9 cuda-nsight-compute-12-9 cuda-nsight-systems-12-9
fi
nvcc --version
nsys --version
ncu --version

if [[ ! -e $lab_dir ]]; then
    checkout="$stage/checkout"
    git init -q "$checkout"
    git -C "$checkout" fetch -q "$stage/source.bundle" HEAD
    test "$(git -C "$checkout" rev-parse FETCH_HEAD)" = "$revision"
    git -C "$checkout" switch -q -c main FETCH_HEAD
    git -C "$checkout" remote add origin https://github.com/vickgidwani/realtime-beamforming-lab.git
    mv -T --no-clobber "$checkout" "$lab_dir"
    test ! -e "$checkout"
fi
cd "$lab_dir"
if [[ $(git rev-parse HEAD) != "$revision" ]]; then
    printf 'Using existing remote checkout at %s (local checkout is %s).\n' "$(git rev-parse HEAD)" "$revision"
fi
pwd
git rev-parse HEAD
git status --short
python3 -m venv .venv
python=.venv/bin/python
"$python" -m pip install --no-cache-dir -e '.[gpu]' 'mach-beamform==0.2.1' 'pymust==0.1.9' \
    'Cython>=3,<4' setuptools pandas psutil matplotlib ipykernel
"$python" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0'
"$python" -m pip install --no-cache-dir 'onnxruntime-gpu[cuda,cudnn]==1.32.0' onnx onnxscript 'transformers>=4.45,<6' pillow
"$python" -m pip check

mkdir -p artifacts/provisioning/inference_lab artifacts/exercise7
cp "$stage/"{__init__,export_model,model}.py artifacts/provisioning/inference_lab/
sha256sum artifacts/provisioning/inference_lab/{export_model,model}.py > artifacts/provisioning/export-source.sha256
"$python" -m pip freeze > artifacts/provisioning/pip-freeze.txt
nvidia-smi -q > artifacts/provisioning/nvidia-smi.txt
PYTHONPATH="$PWD/artifacts/provisioning" "$python" -m inference_lab.export_model \
    --embedding-output artifacts/exercise7/resnet50-embedding.onnx

"$python" - <<'PY'
import cupy as cp
import numpy as np
import onnxruntime as ort
from beamforming_lab.picmus import load_picmus_workload
from beamforming_lab.streaming import load_cython_scheduler
from beamforming_lab.workload import load_pymust_doppler_workload

load_cython_scheduler()
pymust = load_pymust_doppler_workload()
picmus = load_picmus_workload()
print('PyMUST IQ shape:', pymust.iq_elements_samples_frames.shape, flush=True)
print('PICMUS scan shape:', picmus.grid_shape, flush=True)

ort.preload_dlls(directory='')
stream = cp.cuda.Stream(non_blocking=True)
session = ort.InferenceSession(
    'artifacts/exercise7/resnet50-embedding.onnx',
    providers=[('CUDAExecutionProvider', {'device_id': 0, 'user_compute_stream': str(stream.ptr)})],
)
if session.get_providers()[0] != 'CUDAExecutionProvider':
    raise RuntimeError('ONNX Runtime failed to initialize the CUDA provider')
with stream:
    image = cp.zeros((1, 3, 224, 224), dtype=cp.float32)
    embedding = cp.empty((1, 2048), dtype=cp.float32)
    binding = session.io_binding()
    binding.bind_input('pixel_values', 'cuda', 0, np.float32, image.shape, image.data.ptr)
    binding.bind_output('embedding', 'cuda', 0, np.float32, embedding.shape, embedding.data.ptr)
    session.run_with_iobinding(binding)
    if not bool(cp.isfinite(embedding).all()):
        raise RuntimeError('Embedding contains non-finite values')
print('Exercise 7 CUDA I/O Binding smoke check passed:', embedding.shape, flush=True)
PY

.venv/bin/beamform-lab benchmark --dataset pymust --mode cupy --runs 1 --warmup 1 --batch-size 32 \
    --output artifacts/provisioning/pymust-smoke.json
.venv/bin/beamform-lab benchmark --dataset picmus --mode cupy --runs 1 --warmup 1 \
    --output artifacts/provisioning/picmus-smoke.json

if [[ $patched_mach == true ]]; then
    mach_source="$PWD/artifacts/mach-v0.2.1"
    if [[ ! -d $mach_source/.git ]]; then
        git clone --branch v0.2.1 --depth 1 https://github.com/Forest-Neurotech/mach.git "$mach_source"
    fi
    test "$(git -C "$mach_source" rev-parse HEAD)" = 237d38158bf299f67b46d71eaba2cdeac9c6f02a
    mach_patches=("$stage"/mach-v0.2.1-{explicit-stream-async,submission-nvtx,wrapper-nvtx,prepare-beamform}.patch)
    next_patch=0
    # Later patches change context lines in earlier patches. Check newest first.
    for ((index=${#mach_patches[@]}-1; index>=0; index--)); do
        if git -C "$mach_source" apply --reverse --check "${mach_patches[index]}" 2>/dev/null; then
            next_patch=$((index+1))
            break
        fi
    done
    for ((index=next_patch; index<${#mach_patches[@]}; index++)); do
        git -C "$mach_source" apply "${mach_patches[index]}"
    done
    mkdir -p artifacts/mach-async
    wheel_dir=$(mktemp -d "$PWD/artifacts/mach-async/wheels.XXXXXXXX")
    CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=89' "$python" -m pip wheel --no-deps --wheel-dir "$wheel_dir" "$mach_source"
    "$python" -m pip install --no-deps --upgrade --target artifacts/mach-async "$wheel_dir"/*.whl
    PYTHONPATH="$PWD/artifacts/mach-async" "$python" - <<'PY'
from inspect import signature
from mach.kernel import beamform, prepare_beamform
assert callable(prepare_beamform)
assert {'stream', 'synchronize', 'trace_submission'} <= set(signature(beamform).parameters)
PY
fi
printf 'Lab ready in %s\n' "$lab_dir"
REMOTE
