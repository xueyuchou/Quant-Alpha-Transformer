import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch

def evaluate_model(model, test_loader, device):
    """
    运行测试集推理，收集预测值与真实标签
    """
    model.eval()
    predictions = []
    y_trues = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            preds = model(batch_X.to(device)).cpu().numpy().flatten()
            predictions.extend(preds)
            y_trues.extend(batch_y.numpy().flatten())
            
    return predictions, y_trues

def compute_and_plot_ic(predictions, y_trues, test_dates):
    """
    计算逐日截面 Rank IC，打印回测报告，并绘制累计 IC 曲线
    """
    df_eval = pd.DataFrame({
        'date': test_dates,
        'y_pred': predictions,
        'y_true': y_trues
    })

    def calculate_rank_ic(group):
        if len(group) < 2: return np.nan
        corr, _ = stats.spearmanr(group['y_pred'], group['y_true'])
        return corr

    # include_groups=False 抹除 pandas 弃用警告
    daily_ic = df_eval.groupby('date').apply(calculate_rank_ic, include_groups=False).dropna()

    # 打印评估报告
    print("\n" + "="*30)
    print("样本外测试集评估报告")
    print("="*30)
    print(f"平均 Rank IC : {daily_ic.mean():.4f}")
    print(f"IC IR (信息比率) : {daily_ic.mean() / daily_ic.std() if daily_ic.std() != 0 else 0:.4f}")
    print(f"IC 胜率 : {(daily_ic > 0).sum() / len(daily_ic)*100:.2f}%")
    print("="*30)

    # 绘制累计曲线
    plt.figure(figsize=(10, 5))
    plt.plot(daily_ic.cumsum().index, daily_ic.cumsum().values, color='red', linewidth=2)
    plt.title('Cumulative Rank IC (Out-of-Sample Test)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Rank IC')
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.show()