import matplotlib.pyplot as plt

def plot_trades(data, buy_steps, sell_steps, portfolio_values=None, title="Trading Visualization", save_path=None):
    """
    data: 가격 데이터 (list or np.array)
    buy_steps: 매수한 step 인덱스 리스트
    sell_steps: 매도한 step 인덱스 리스트
    portfolio_values: 각 step별 포트폴리오 가치 (list, optional)
    save_path: 저장할 파일 경로 (str, optional)
    """
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # 왼쪽 y축: 주가
    ax1.plot(data, label="Price", color="blue")
    ax1.scatter(buy_steps, [data[i] for i in buy_steps], color="green", marker="^", label="Buy", s=100, zorder=5)
    ax1.scatter(sell_steps, [data[i] for i in sell_steps], color="red", marker="v", label="Sell", s=100, zorder=5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Stock Price", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)

    # 오른쪽 y축: 포트폴리오 가치
    if portfolio_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(portfolio_values, label="Portfolio Value", color="orange", linestyle="--")
        ax2.set_ylabel("Portfolio Value", color="orange")
        ax2.tick_params(axis='y', labelcolor="orange")
        # 범례 합치기
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
