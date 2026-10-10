#!/usr/bin/env bash
# Drive the phase-0 cloud runs from the PC (inside WSL), with the official Colab CLI.
#   bash colab_phase0.sh up <plan> <gpu>   allocate the VM if needed, upload code + local checkpoints, start the launcher
#   bash colab_phase0.sh status            jobs state and the end of the launcher log
#   bash colab_phase0.sh diagnose          read-only launcher, downloads and resource diagnostics
#   bash colab_phase0.sh pull              download cloud results into phase0/results (safe, never goes backwards)
#   bash colab_phase0.sh down              release the VM (stops compute-unit usage)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
S=phase0
REPO="${DLLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # repository root (this file: phase0/colab/)
REMOTE=/content/dllm/phase0/results
COLAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Every official CLI stream is sanitized before it can reach a terminal or campaign journal.
colab() { python3 "$COLAB_DIR/safe_cli.py" "$@"; }

# Keep one open-file-description lock across helpers, transfer and bootstrap; nested reads do not lock.
case "${1:-}" in up|resume-up|down)
  receipt="${DLLM_CAMPAIGN_RECEIPT:-$HOME/.local/state/myriad-colab/ownership.json}"
  mkdir -p "$(dirname "$receipt")"
  if [ -z "${DLLM_LIFECYCLE_LOCK_FD:-}" ]; then
    exec 9>>"$receipt.lock"
    flock -n 9 || { echo 'another lifecycle operation holds the lock'; exit 73; }
    export DLLM_LIFECYCLE_LOCK_FD=9
  fi  # Helpers validate and share a lock explicitly inherited from cleanup-only.
esac

checkpoint_allowed() {
  local family="${1%%_*}"
  [[ "$families" = *" $family "* ]]
}

case "${1:-}" in
  resume-up)
    plan="${2:?plan}"; gpu="${3:?original gpu}"
    DLLM_OPERATION_IDENTITY=$(python3 "$COLAB_DIR/session_json.py" identity)
    export DLLM_OPERATION_IDENTITY
    archive=$(python3 "$COLAB_DIR/session_json.py" resume "$plan" "$gpu")
    export DLLM_REQUIRE_OWNER=1
    snapshot=$(bash "$REPO/phase0/colab/colab_phase0.sh" snapshot)
    printf '%s' "$snapshot" | python3 "$COLAB_DIR/session_json.py" guard
    python3 "$COLAB_DIR/transfer.py" "$archive"
    colab exec -s "$S" --timeout 1800 -f "$archive.colab_bootstrap.py"
    ;;
  usage|usage-json)
    python3 "$COLAB_DIR/session_json.py" usage
    ;;
  sessions-json)
    sessions_output=$(colab sessions)
    printf '%s\n' "$sessions_output" | python3 "$REPO/phase0/colab/session_json.py"
    ;;
  snapshot)
    if [ "${DLLM_SUPERVISED:-}" = 1 ]; then
      colab exec -s "$S" --timeout 60 -f "$REPO/phase0/colab/snapshot.py"
    else
      python3 "$REPO/phase0/colab/read_timeout.py" python3 "$COLAB_DIR/safe_cli.py" \
        exec -s "$S" --timeout 60 -f "$REPO/phase0/colab/snapshot.py"
    fi
    ;;
  up)
    plan="${2:?plan}"; gpu="${3:-auto}"
    if [ "${DLLM_SUPERVISED:-}" = 1 ] && [ "$plan" != aa-1 ]; then
      echo 'exclusive supervision requires aa-1'; exit 2
    fi
    export DLLM_CAMPAIGN_ID="${DLLM_CAMPAIGN_ID:-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')}"
    if [ "$gpu" = auto ]; then gpu=$(python3 "$COLAB_DIR/session_json.py" select-gpu "$plan"); fi
    python3 "$COLAB_DIR/session_json.py" allocate "$plan" "$gpu"
    DLLM_OPERATION_IDENTITY=$(python3 "$COLAB_DIR/session_json.py" identity)
    export DLLM_OPERATION_IDENTITY
    families=$(python3 "$COLAB_DIR/session_json.py" families "$plan")
    export DLLM_REQUIRE_OWNER=1
    snapshot=$(bash "$REPO/phase0/colab/colab_phase0.sh" snapshot)
    printf '%s' "$snapshot" | python3 "$COLAB_DIR/session_json.py" guard
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/stage/checkpoints"
    tar cf - -C "$REPO" --exclude='phase0/.venv' --exclude='phase0/results' --exclude='phase0/data' \
        --exclude='__pycache__' --exclude='*.h5' --exclude='.cache' phase0 | tar xf - -C "$tmp/stage"
    case "$plan" in aa-*)  # sent to the VM only (GPQA terms): phase0/data/ is ignored by git and left out of the other plans
      mkdir -p "$tmp/stage/phase0/data"
      if [ -e "$REPO/phase0/data/gpqa_diamond.csv" ]; then
        cp "$REPO/phase0/data/gpqa_diamond.csv" "$tmp/stage/phase0/data/"
      fi ;;
    esac
    echo "$plan" > "$tmp/stage/phase0/colab_plan.txt"
    printf '%s\n' "${DLLM_CAMPAIGN_ID:?campaign id}" > "$tmp/stage/phase0/colab_campaign_id.txt"
    if [ "$plan" = aa-1 ]; then
      python3 "$COLAB_DIR/provenance.py" "$tmp/stage"
      mkdir -p "$REPO/phase0/results"
      cp "$tmp/stage/phase0/results/e12_campaign_sources.json" "$REPO/phase0/results/e12_campaign_sources.local.json"
    fi
    # local copies of earlier cloud results (checkpoints), so a new VM resumes instead of restarting
    for f in "$REPO"/phase0/results/mc_*_colab_*.jsonl "$REPO"/phase0/results/gen_*_colab_*.jsonl "$REPO"/phase0/results/solo_*_colab_*.jsonl "$REPO"/phase0/results/sot_*_colab_*.jsonl \
             "$REPO"/phase0/results/code_*_colab_*.jsonl "$REPO"/phase0/results/codeexec_*_colab_*.jsonl \
             "$REPO"/phase0/results/aa_*_colab_*.jsonl "$REPO"/phase0/results/sciexec_*_colab_*.jsonl; do
      [[ "$f" = *.timing.jsonl ]] && continue
      checkpoint_allowed "$(basename "$f")" || continue
      if [ -e "$f" ] && [ -e "$f.meta.json" ]; then
        if [ "$plan" = aa-1 ]; then
          for checkpoint in "$f" "$f.meta.json" "$f.timing.jsonl"; do
            [ ! -e "$checkpoint" ] || [ "$(stat -c %s "$checkpoint")" -le 8388608 ] || {
              echo 'E12 checkpoint exceeds 8 MiB; upload refused'; exit 65;
            }
          done
        fi
        cp "$f" "$f.meta.json" "$tmp/stage/checkpoints/"
        if [[ "$f" = */aa_* ]] && [ -e "$f.timing.jsonl" ]; then cp "$f.timing.jsonl" "$tmp/stage/checkpoints/"; fi
      fi
    done
    tar czf "$tmp/dllm.tgz" -C "$tmp/stage" phase0 checkpoints
    archive=$(python3 "$COLAB_DIR/session_json.py" archive "$tmp/dllm.tgz" "$plan" \
      "$REPO/phase0/colab_bootstrap.py")
    python3 "$COLAB_DIR/transfer.py" "$archive"
    rm -rf "$tmp"
    code=0
    colab exec -s "$S" --timeout 1800 -f "$archive.colab_bootstrap.py" || code=$?
    if [ "$code" = 73 ]; then python3 "$COLAB_DIR/session_json.py" refuse; fi
    exit "$code"
    ;;
  status)
    # The CLI's 60s limit covers remote execution, not a stuck local connection. Bound the whole process group.
    printf '%s\n' "import os, subprocess" \
      "for p in ('$REMOTE/colab_bootstrap_status.json', '$REMOTE/colab_status.json'):" \
      "    print(open(p).read() if os.path.exists(p) else f'{os.path.basename(p)} : absent')" \
      "print(subprocess.run(['tail','-5','$REMOTE/colab_launcher.log'],capture_output=True,text=True).stdout)" \
      "print(subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)" \
      | python3 "$REPO/phase0/colab/read_timeout.py" python3 "$COLAB_DIR/safe_cli.py" exec -s "$S" --timeout 60
    ;;
  diagnose)
    python3 "$REPO/phase0/colab/read_timeout.py" python3 "$COLAB_DIR/safe_cli.py" \
        exec -s "$S" --timeout 60 -f "$REPO/phase0/colab/diagnose.py"
    ;;
  pull)
    plan=$(python3 "$COLAB_DIR/session_json.py" plan)
    DLLM_OPERATION_IDENTITY=$(python3 "$COLAB_DIR/session_json.py" identity)
    export DLLM_OPERATION_IDENTITY
    families=$(python3 "$COLAB_DIR/session_json.py" families "$plan")
    export DLLM_REQUIRE_OWNER=1
    dest="$REPO/phase0/results"
    mkdir -p "$dest"
    stage=$(mktemp -d)
    listing=$(printf '%s\n' "import glob, os" \
      "print('LISTE-OK')" \
      "print('\n'.join(os.path.basename(f) for f in sorted(glob.glob('$REMOTE/mc_*_colab_*.jsonl') + glob.glob('$REMOTE/gen_*_colab_*.jsonl') + glob.glob('$REMOTE/solo_*_colab_*.jsonl') + glob.glob('$REMOTE/sot_*_colab_*.jsonl') + glob.glob('$REMOTE/code_*_colab_*.jsonl') + glob.glob('$REMOTE/codeexec_*_colab_*.jsonl') + glob.glob('$REMOTE/aa_*_colab_*.jsonl') + glob.glob('$REMOTE/sciexec_*_colab_*.jsonl'))))" \
      | colab exec -s "$S" --timeout 60 | tr -d '\r')
    echo "$listing" | grep -q '^LISTE-OK$' || { echo "listing impossible"; exit 1; }
    status=0
    if [ "$plan" = aa-1 ]; then
      for f in e12_campaign_sources.json colab_status.json colab_bootstrap_status.json; do
        if colab download -s "$S" "$REMOTE/$f" "$stage/$f"; then
          python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$stage/$f"
        else status=1; fi
      done
      [ "$status" = 0 ] || { rm -rf "$stage"; exit "$status"; }
      python3 - "$stage" "$DLLM_OPERATION_IDENTITY" <<'PY'
import json, sys
from pathlib import Path
stage = Path(sys.argv[1])
boot = json.loads((stage / "colab_bootstrap_status.json").read_text(encoding="utf-8"))
if boot.get("campaign_id") != json.loads(sys.argv[2])["campaign_id"]:
    raise SystemExit("foreign recovered campaign: no local metadata publication authorized")
PY
      cp "$stage/e12_campaign_sources.json" "$stage/colab_status.json" "$stage/colab_bootstrap_status.json" "$dest/"
    fi
    for f in $(echo "$listing" | grep -E '^(mc|gen|solo|sot|code|codeexec|aa|sciexec)_.*_colab_.*\.jsonl$'); do
      [[ "$f" = *.timing.jsonl ]] && continue
      checkpoint_allowed "$f" || continue
      if colab download -s "$S" "$REMOTE/$f" "$stage/$f" >/dev/null && \
         colab download -s "$S" "$REMOTE/$f.meta.json" "$stage/$f.meta.json" >/dev/null; then
        # validate the pair, never go backwards, publish atomically, keep the previous checkpoint
        if python3 "$REPO/phase0/colab/checkpoint.py" "$stage/$f" "$dest/$f"; then echo "récupéré $f"; else status=1; fi
        if [[ "$f" = aa_* ]] && python3 -c \
          'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("measurements") else 1)' \
          "$stage/$f.meta.json"; then
          if colab download -s "$S" "$REMOTE/$f.timing.jsonl" "$stage/$f.timing.jsonl"; then
            python3 "$REPO/phase0/aa_timing.py" "$stage/$f.timing.jsonl" "$dest/$f.timing.jsonl" \
              "$stage/$f.meta.json" || status=1
          else status=1; fi
        fi
      else
        echo "ÉCHEC téléchargement $f"; status=1
      fi
    done
    if [ "${2:-}" = final ] && [ "$plan" = aa-1 ]; then
      python3 "$REPO/phase0/colab/verify_e12_pull.py" "$stage" \
        "$REPO/phase0/results/e12_campaign_sources.local.json" || status=1
    fi
    rm -rf "$stage"
    exit $status
    ;;
  down)
    python3 "$COLAB_DIR/session_json.py" down
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
