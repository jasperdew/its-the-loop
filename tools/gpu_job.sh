#!/usr/bin/env bash
# Run one experiment on a throwaway L4 VM: upload src/, run a command, save results/ to GCS, self-delete.
# No browser, no Colab. Results land in gs://$BUCKET/$NAME/results.
#
# Usage: tools/gpu_job.sh NAME "python promptvariants.py --out results" [ZONE]
# Example: tools/gpu_job.sh 02_survey "for m in qwen3.5-0.8b qwen3.5-2b; do python loopsurvey.py --model \$m; done"
set -euo pipefail
NAME=$1; CMD=$2; ZONE=${3:-asia-northeast3-a}
PROJECT=${PROJECT:?set PROJECT to your GCP project}
BUCKET=${BUCKET:?set BUCKET to a GCS bucket you own}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VM="job-${NAME//_/-}"

gcloud storage rsync -r "$ROOT/src" "gs://$BUCKET/$NAME/src" --quiet --exclude=".*__pycache__.*"

cat > /tmp/startup_$NAME.sh <<EOF
#!/bin/bash
exec > /var/log/job.log 2>&1
set -x
upload() { gsutil -m cp -r /job/results /var/log/job.log gs://$BUCKET/$NAME/ || true; }
trap 'upload; gcloud compute instances delete $VM --zone=$ZONE --quiet' EXIT
mkdir -p /job && cd /job
gsutil -m cp -r gs://$BUCKET/$NAME/src/* .
export HOME=/root
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq g++ build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:\$PATH
uv venv -p 3.12 /job/venv
export PATH=/job/venv/bin:\$PATH VIRTUAL_ENV=/job/venv
uv pip install "vllm==0.33.0" "math-verify[antlr4_13_2]==0.9.0" datasets pandas
uv pip uninstall torchaudio || true
nvidia-smi
mkdir -p results
( while sleep 300; do upload; done ) &
$CMD
EOF

gcloud compute instances create "$VM" --project=$PROJECT --zone="$ZONE" \
  --machine-type=g2-standard-4 --maintenance-policy=TERMINATE \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --scopes=cloud-platform \
  --metadata=install-nvidia-driver=True \
  --metadata-from-file=startup-script=/tmp/startup_$NAME.sh
echo "started $VM in $ZONE; log: gs://$BUCKET/$NAME/job.log (every 5 min)"
