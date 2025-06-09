#!/usr/bin/env bash
# make_detailed_logs.sh
# 모든 알고리즘(ac, a2c, …, pg)×모든 데이터셋×모든 reward 조합에 대해
# generate_eval_log.py 를 실행해 logs_detailed/<algo>/…csv 를 만든다.

set -euo pipefail

PY=detailed_log.py                 # 파이썬 스크립트 이름
ALGOS=(ac a2c a3c deepsarsa dqn pg)     # 알고리즘 폴더 목록
REWARDS='cvar|kelly|none|sharpe|var'    # 정규식 패턴

for algo in "${ALGOS[@]}"; do
    src_dir="./logs/${algo}"
    dst_dir="./logs_detailed/${algo}"
    mkdir -p "${dst_dir}"

    # logs/<algo> 안의 모든 *.txt 순회
    for txt_path in "${src_dir}"/*.txt; do
        [[ -e "$txt_path" ]] || continue          # 파일 없으면 skip
        txt_file=$(basename "$txt_path")          # ex) hangsen_ours_fold1_cvar.txt
        base_name="${txt_file%.txt}"              # 확장자 제거

        # 데이터셋 basename (reward suffix 제거) → ex) hangsen_ours_fold1
        dataset_base=$(echo "$base_name" | sed -E "s/_(${REWARDS})$//")
        csv_path="./data/${dataset_base}.csv"

        # CSV 없으면 경고만 출력
        if [[ ! -f "$csv_path" ]]; then
            echo "⚠️  CSV not found: $csv_path  (skip)" >&2
            continue
        fi

        out_path="${dst_dir}/${base_name}.csv"    # 결과 파일
        echo "▶️  $algo : $(basename "$txt_file" .txt) → $(basename "$out_path")"
        python "$PY" "$csv_path" "$txt_path" "$out_path"
    done
done
