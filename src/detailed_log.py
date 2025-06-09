#!/usr/bin/env python3
# generate_log.py
import csv
import sys
from collections import defaultdict

def generate_log(csv_path: str, txt_path: str, output_path: str) -> None:
    # ────────────── 1. 주가 로드 ──────────────
    prices = []
    with open(csv_path, newline='', encoding='utf-8') as fcsv:
        rdr = csv.reader(fcsv)
        for row in rdr:
            if not row:                 # 빈 줄 방지
                continue
            # 5번째 열(인덱스 4)을 실가격으로 사용
            price = float(row[4].replace(',', '').strip())
            prices.append(price)

    prices = prices[1:]  # 첫 줄은 헤더이므로 제외
    split_idx = int(len(prices) * 0.8)
    prices = prices[split_idx:]  # 20%만 사용

    # ────────────── 2. 초기 현금·행동 파싱 ──────────────
    actions_by_day = defaultdict(list)
    initial_cash = None

    with open(txt_path, encoding='utf-8') as ftxt:
        for line in ftxt:
            line = line.strip()
            if line.startswith('DAY'):
                # 예: DAY 133 | PRICE $19645.57 | ACTION SELL
                parts = [p.strip() for p in line.split('|')]
                day   = int(parts[0].split()[1])
                act   = parts[2].split()[1].upper()   # BUY / SELL / LIQUIDATE
                actions_by_day[day].append(act)
            elif line.startswith('Initial Cash'):
                # 예: Initial Cash: $169335.20
                initial_cash = float(line.split('$')[1].replace(',', ''))

    if initial_cash is None:
        raise ValueError('txt 파일에서 Initial Cash 를 찾을 수 없습니다.')

    # ────────────── 3. 일별 시뮬레이션 & 로그 작성 ──────────────
    cash   = initial_cash
    shares = 0

    with open(output_path, 'w', newline='', encoding='utf-8') as fout:
        w = csv.writer(fout)
        w.writerow(['DAY', 'Price', 'Cash', 'Shares', 'PortfolioValue', 'Action'])

        for day, price in enumerate(prices):
            acts = actions_by_day.get(day, ['HOLD'])

            # 모든 행동을 동일한 날짜의 동일 가격에 수행
            for act in acts:
                if act == 'BUY' and cash >= price:
                    cash   -= price
                    shares += 1
                elif act in ('SELL', 'LIQUIDATE') and shares > 0:
                    cash   += price
                    shares -= 1
                # HOLD 는 상태 변화 없음

            port_val = cash + shares * price
            w.writerow([
                day,
                f'{price:.2f}',
                f'{cash:.2f}',
                shares,
                f'{port_val:.2f}',
                ';'.join(acts)            # 복수 행동 시 “BUY;SELL” 등으로 표기
            ])

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:  python generate_log.py <prices.csv> <actions.txt> <output.csv>')
        sys.exit(1)
    generate_log(*sys.argv[1:])
