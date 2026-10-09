#!/usr/bin/env bash
# Drive the phase-0 cloud runs from the PC (inside WSL), with the official Colab CLI.
#   bash colab_phase0.sh up <plan> <gpu>   allocate the VM if needed, upload code + local checkpoints, start the launcher
#   bash colab_phase0.sh status            jobs state and the end of the launcher log
#   bash colab_phase0.sh pull              download cloud results into phase0/results (safe, never goes backwards)
#   bash colab_phase0.sh down              release the VM (stops compute-unit usage)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
S=phase0
REPO="${DLLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # repository root (this file: phase0/colab/)
REMOTE=/content/dllm/phase0/results

case "${1:-}" in
  up)
    plan="${2:?plan}"; gpu="${3:?gpu}"
    colab sessions 2>/dev/null | grep -q "$S" || colab new -s "$S" --gpu "$gpu"
    tmp=$(mktemp -d)
    mkdir -p "$tmp/stage/checkpoints"
    tar cf - -C "$REPO" --exclude='phase0/.venv' --exclude='phase0/results' --exclude='phase0/data' \
        --exclude='__pycache__' phase0 | tar xf - -C "$tmp/stage"
    echo "$plan" > "$tmp/stage/phase0/colab_plan.txt"
    # local copies of earlier cloud results (checkpoints), so a new VM resumes instead of restarting
    for f in "$REPO"/phase0/results/mc_*_colab_*.jsonl "$REPO"/phase0/results/gen_*_colab_*.jsonl "$REPO"/phase0/results/solo_*_colab_*.jsonl "$REPO"/phase0/results/sot_*_colab_*.jsonl; do
      [ -e "$f" ] && [ -e "$f.meta.json" ] && cp "$f" "$f.meta.json" "$tmp/stage/checkpoints/"
    done
    tar czf "$tmp/dllm.tgz" -C "$tmp/stage" phase0 checkpoints
    colab upload -s "$S" "$tmp/dllm.tgz" /content/dllm.tgz
    rm -rf "$tmp"
    colab exec -s "$S" --timeout 1800 -f "$REPO/phase0/colab_bootstrap.py"
    ;;
  status)
    printf '%s\n' "import os, subprocess" \
      "for p in ('$REMOTE/colab_bootstrap_status.json', '$REMOTE/colab_status.json'):" \
      "    print(open(p).read() if os.path.exists(p) else f'{os.path.basename(p)} : absent')" \
      "print(subprocess.run(['tail','-5','$REMOTE/colab_launcher.log'],capture_output=True,text=True).stdout)" \
      "print(subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)" \
      | colab exec -s "$S" --timeout 60
    ;;
  pull)
    dest="$REPO/phase0/results"
    stage=$(mktemp -d)
    listing=$(printf '%s\n' "import glob, os" \
      "print('LISTE-OK')" \
      "print('\n'.join(os.path.basename(f) for f in sorted(glob.glob('$REMOTE/mc_*_colab_*.jsonl') + glob.glob('$REMOTE/gen_*_colab_*.jsonl') + glob.glob('$REMOTE/solo_*_colab_*.jsonl') + glob.glob('$REMOTE/sot_*_colab_*.jsonl'))))" \
      | colab exec -s "$S" --timeout 60 | tr -d '\r')
    echo "$listing" | grep -q '^LISTE-OK$' || { echo "listing impossible"; exit 1; }
    status=0
    for f in $(echo "$listing" | grep -E '^(mc|gen|solo|sot)_.*_colab_.*\.jsonl$'); do
      if colab download -s "$S" "$REMOTE/$f" "$stage/$f" >/dev/null && \
         colab download -s "$S" "$REMOTE/$f.meta.json" "$stage/$f.meta.json" >/dev/null; then
        # validate the pair, never go backwards, publish atomically, keep the previous checkpoint
        if python3 "$REPO/phase0/colab/checkpoint.py" "$stage/$f" "$dest/$f"; then echo "récupéré $f"; else status=1; fi
      else
        echo "ÉCHEC téléchargement $f"; status=1
      fi
    done
    rm -rf "$stage"
    exit $status
    ;;
  down)
    colab stop -s "$S"
    ;;
  *)
    sed -n '2,7p' "$0"; exit 1 ;;
esac
