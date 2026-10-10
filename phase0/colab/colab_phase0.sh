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

case "${1:-}" in
  sessions-json)
    sessions_output=$(colab sessions)
    printf '%s\n' "$sessions_output" | python3 "$REPO/phase0/colab/session_json.py"
    ;;
  snapshot)
    if [ "${DLLM_SUPERVISED:-}" = 1 ]; then
      colab exec -s "$S" --timeout 60 -f "$REPO/phase0/colab/snapshot.py"
    else
      python3 "$REPO/phase0/colab/read_timeout.py" colab exec -s "$S" --timeout 60 -f "$REPO/phase0/colab/snapshot.py"
    fi
    ;;
  up)
    plan="${2:?plan}"; gpu="${3:?gpu}"
    case "$plan" in aa-*)  # E12: SciCode targets required, GPQA optional, checked before a VM is allocated
      for f in scicode_test_data.h5; do
        [ -e "$REPO/phase0/data/$f" ] || { echo "manque phase0/data/$f : voir la section E12 de phase0/README.md (étapes manuelles)"; exit 1; }
      done ;;
    esac
    if [ -n "${DLLM_CAMPAIGN_RECEIPT:-}" ]; then
      [ "$plan" = aa-1 ] || { echo 'exclusive supervision requires aa-1'; exit 2; }
      sessions=$(bash "$REPO/phase0/colab/colab_phase0.sh" sessions-json)
      printf '%s' "$sessions" | python3 -c 'import json,sys; sys.exit(73 if json.load(sys.stdin)["phase0"] else 0)'
      colab new -s "$S" --gpu "$gpu"
      # A receipt is written only after our allocation succeeded; failures before it never authorize down.
      bash "$REPO/phase0/colab/colab_phase0.sh" sessions-json > "$DLLM_CAMPAIGN_RECEIPT"
      python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["phase0"]; assert s and s["hardware"] == "A100"' \
        "$DLLM_CAMPAIGN_RECEIPT"
      snapshot=$(bash "$REPO/phase0/colab/colab_phase0.sh" snapshot)
      code=0
      printf '%s' "$snapshot" | python3 -c 'import json,sys
s=json.load(sys.stdin)
assert s.get("schema") == 1 and type(s.get("campaign_present")) is bool
sys.exit(73 if s["campaign_present"] else 0)' || code=$?
      if [ "$code" = 73 ]; then
        rm -f "$DLLM_CAMPAIGN_RECEIPT"  # unexpected existing campaign: do not touch it, including cleanup
        exit 73
      fi
      [ "$code" = 0 ] || exit "$code"
    else
      colab sessions 2>/dev/null | grep -q "$S" || colab new -s "$S" --gpu "$gpu"
    fi
    tmp=$(mktemp -d)
    mkdir -p "$tmp/stage/checkpoints"
    tar cf - -C "$REPO" --exclude='phase0/.venv' --exclude='phase0/results' --exclude='phase0/data' \
        --exclude='__pycache__' phase0 | tar xf - -C "$tmp/stage"
    case "$plan" in aa-*)  # sent to the VM only (GPQA terms): phase0/data/ is ignored by git and left out of the other plans
      mkdir -p "$tmp/stage/phase0/data"
      cp "$REPO/phase0/data/scicode_test_data.h5" "$tmp/stage/phase0/data/"
      if [ -e "$REPO/phase0/data/gpqa_diamond.csv" ]; then
        cp "$REPO/phase0/data/gpqa_diamond.csv" "$tmp/stage/phase0/data/"
      fi ;;
    esac
    echo "$plan" > "$tmp/stage/phase0/colab_plan.txt"
    if [ -n "${DLLM_CAMPAIGN_RECEIPT:-}" ]; then
      printf '%s\n' "${DLLM_CAMPAIGN_ID:?campaign id}" > "$tmp/stage/phase0/colab_campaign_id.txt"
    fi
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
      if [ -n "${DLLM_CAMPAIGN_RECEIPT:-}" ] && [[ "$f" != */aa_* && "$f" != */sciexec_* ]]; then continue; fi
      if [ -e "$f" ] && [ -e "$f.meta.json" ]; then
        cp "$f" "$f.meta.json" "$tmp/stage/checkpoints/"
        if [[ "$f" = */aa_* ]] && [ -e "$f.timing.jsonl" ]; then cp "$f.timing.jsonl" "$tmp/stage/checkpoints/"; fi
      fi
    done
    tar czf "$tmp/dllm.tgz" -C "$tmp/stage" phase0 checkpoints
    colab upload -s "$S" "$tmp/dllm.tgz" /content/dllm.tgz
    rm -rf "$tmp"
    code=0
    colab exec -s "$S" --timeout 1800 -f "$REPO/phase0/colab_bootstrap.py" || code=$?
    if [ "$code" = 73 ] && [ -n "${DLLM_CAMPAIGN_RECEIPT:-}" ]; then rm -f "$DLLM_CAMPAIGN_RECEIPT"; fi
    exit "$code"
    ;;
  status)
    # The CLI's 60s limit covers remote execution, not a stuck local connection. Bound the whole process group.
    printf '%s\n' "import os, subprocess" \
      "for p in ('$REMOTE/colab_bootstrap_status.json', '$REMOTE/colab_status.json'):" \
      "    print(open(p).read() if os.path.exists(p) else f'{os.path.basename(p)} : absent')" \
      "print(subprocess.run(['tail','-5','$REMOTE/colab_launcher.log'],capture_output=True,text=True).stdout)" \
      "print(subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)" \
      | python3 "$REPO/phase0/colab/read_timeout.py" colab exec -s "$S" --timeout 60
    ;;
  diagnose)
    python3 "$REPO/phase0/colab/read_timeout.py" colab exec -s "$S" --timeout 60 -f "$REPO/phase0/colab/diagnose.py"
    ;;
  pull)
    dest="$REPO/phase0/results"
    mkdir -p "$dest"
    stage=$(mktemp -d)
    listing=$(printf '%s\n' "import glob, os" \
      "print('LISTE-OK')" \
      "print('\n'.join(os.path.basename(f) for f in sorted(glob.glob('$REMOTE/mc_*_colab_*.jsonl') + glob.glob('$REMOTE/gen_*_colab_*.jsonl') + glob.glob('$REMOTE/solo_*_colab_*.jsonl') + glob.glob('$REMOTE/sot_*_colab_*.jsonl') + glob.glob('$REMOTE/code_*_colab_*.jsonl') + glob.glob('$REMOTE/codeexec_*_colab_*.jsonl') + glob.glob('$REMOTE/aa_*_colab_*.jsonl') + glob.glob('$REMOTE/sciexec_*_colab_*.jsonl'))))" \
      | colab exec -s "$S" --timeout 60 | tr -d '\r')
    echo "$listing" | grep -q '^LISTE-OK$' || { echo "listing impossible"; exit 1; }
    status=0
    if [ -n "${DLLM_CAMPAIGN_RECEIPT:-}" ]; then
      for f in e12_campaign_sources.json colab_status.json colab_bootstrap_status.json; do
        if colab download -s "$S" "$REMOTE/$f" "$stage/$f"; then
          python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$stage/$f"
          mkdir -p "$dest"
          cp "$stage/$f" "$dest/$f"
        else status=1; fi
      done
    fi
    for f in $(echo "$listing" | grep -E '^(mc|gen|solo|sot|code|codeexec|aa|sciexec)_.*_colab_.*\.jsonl$'); do
      [[ "$f" = *.timing.jsonl ]] && continue
      if [ -n "${DLLM_CAMPAIGN_RECEIPT:-}" ] && [[ "$f" != aa_* && "$f" != sciexec_* ]]; then continue; fi
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
    if [ "${2:-}" = final ]; then
      python3 "$REPO/phase0/colab/verify_e12_pull.py" "$stage" \
        "$REPO/phase0/results/e12_campaign_sources.local.json" || status=1
    fi
    rm -rf "$stage"
    exit $status
    ;;
  down)
    colab stop -s "$S"
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
