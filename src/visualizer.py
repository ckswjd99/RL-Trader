import matplotlib.pyplot as plt

def plot_trades(data, buy_steps, sell_steps, portfolio_values=None, title="Trading Visualization", save_path=None):
    """
    data: 가격 데이터 (list or np.array)
    buy_steps: 매수한 step 인덱스 리스트
    sell_steps: 매도한 step 인덱스 리스트
    portfolio_values: 각 step별 포트폴리오 가치 (list, optional)
    save_path: 저장할 파일 경로 (str, optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # 첫 번째 그래프: 주가 + 매수/매도
    ax1.plot(data, label="Price", color="blue")
    ax1.scatter(buy_steps, [data[i] for i in buy_steps], color="green", marker="^", label="Buy", s=100, zorder=5)
    ax1.scatter(sell_steps, [data[i] for i in sell_steps], color="red", marker="v", label="Sell", s=100, zorder=5)
    ax1.set_ylabel("Stock Price")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # 두 번째 그래프: 포트폴리오 가치
    if portfolio_values is not None:
        ax2.plot(portfolio_values, label="Portfolio Value", color="orange", linestyle="--")
        ax2.set_ylabel("Portfolio Value")
        ax2.legend(loc="upper left")
        ax2.grid(True)

    ax2.set_xlabel("Step")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
