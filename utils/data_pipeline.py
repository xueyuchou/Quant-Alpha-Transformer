import pandas as pd
import numpy as np
import torch
import gc
import qlib
from qlib.data import D
from torch.utils.data import Dataset, DataLoader

def load_and_preprocess_data(parquet_file, start_time='2018-01-01', end_time='2022-03-31'):
    """
    加载 Qlib Alpha158 数据，获取未来收益率标签，并进行内存优化与截面中性化清洗。
    """
    # ==========================================
    # 1. 读取特征数据与初始化 Qlib
    # ==========================================
    print(f"--> [1/6] 正在读取特征数据: {parquet_file} ...")
    df_export = pd.read_parquet(parquet_file)
    
    print("--> [2/6] 初始化 Qlib 底层引擎并获取未来收益率标签 ...")
    provider_uri = "/root/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region="cn")

    label_expr = ['Ref($open, -2)/Ref($open, -1) - 1']
    df_label = D.features(
        instruments=D.instruments(market="csi500"),
        start_time=start_time,
        end_time=end_time,
        fields=label_expr
    )
    df_label.columns = ['label_return']
    df_label = df_label.reset_index()
    df_label.rename(columns={'datetime': 'date', 'instrument': 'ticker'}, inplace=True)

    # ==========================================
    # 2. 内存预处理与截面标准化
    # ==========================================
    print("--> [3/6] 合并特征与标签，提取特征列名 ...")
    df_full = pd.merge(df_export, df_label, on=['date', 'ticker'], how='inner')
    feature_cols = [col for col in df_full.columns if col not in ['date', 'ticker', 'label_return']]

    # 核心动作：立即释放未合并前的大表内存
    del df_export, df_label
    gc.collect()

    print("--> [4/6] 降低精度至 float32 并执行前向填充 ...")
    df_full[feature_cols] = df_full[feature_cols].astype(np.float32)
    df_full['label_return'] = df_full['label_return'].astype(np.float32)
    
    df_full = df_full.sort_values(by=['ticker', 'date'])
    df_full[feature_cols] = df_full.groupby('ticker')[feature_cols].ffill().fillna(0)

    print("--> [5/6] 执行特征截面标准化 (Z-score) ...")
    daily_mean_x = df_full.groupby('date')[feature_cols].transform('mean')
    daily_std_x = df_full.groupby('date')[feature_cols].transform('std')
    df_full[feature_cols] = ((df_full[feature_cols] - daily_mean_x) / daily_std_x).fillna(0).astype(np.float32)

    print("--> [6/6] 执行标签极值截断 (MAD/Percentile) 与截面标准化 ...")
    df_full['label_return'] = df_full['label_return'].clip(
        lower=df_full['label_return'].quantile(0.01),
        upper=df_full['label_return'].quantile(0.99)
    )
    daily_mean_y = df_full.groupby('date')['label_return'].transform('mean')
    daily_std_y = df_full.groupby('date')['label_return'].transform('std')
    df_full['label_norm'] = ((df_full['label_return'] - daily_mean_y) / daily_std_y).fillna(0).astype(np.float32)

    # 清理中间变量
    del daily_mean_x, daily_std_x, daily_mean_y, daily_std_y
    gc.collect()
    print("✅ 数据管线 (Pipeline) 清洗完毕！")
    
    # 返回清洗好的大表和特征列名列表，供后续生成 Dataset 使用
    return df_full, feature_cols

def create_time_series_tensors(df_full, feature_cols, seq_len=20):
    """
    通过滑动窗口生成 3D 时序张量，使用物理内存预分配防止 OOM。
    """
    print(f"--> [1/3] 计算张量切片总数 (Seq_Len={seq_len}) ...")
    ticker_counts = df_full.groupby('ticker').size()
    valid_tickers = ticker_counts[ticker_counts > seq_len]
    total_samples = (valid_tickers - seq_len).sum()

    print(f"--> [2/3] 向系统申请 {total_samples} 行的物理连续内存 ...")
    # 规避内存翻倍，直接申请最终大小的空数组
    X_all = np.empty((total_samples, seq_len, len(feature_cols)), dtype=np.float32)
    y_all = np.empty((total_samples,), dtype=np.float32)
    dates_all = np.empty((total_samples,), dtype=object)

    print("--> [3/3] 填装时序滑动窗口 ...")
    idx = 0
    for ticker, group in df_full.groupby('ticker'):
        if len(group) <= seq_len:
            continue
        X_vals = group[feature_cols].values
        y_vals = group['label_norm'].values
        dates = group['date'].values

        n_samples = len(group) - seq_len
        for i in range(n_samples):
            X_all[idx] = X_vals[i : i + seq_len]
            y_all[idx] = y_vals[i + seq_len]
            dates_all[idx] = dates[i + seq_len]
            idx += 1

    # 销毁母表，释放内存
    del df_full
    gc.collect()
    print(f"✅ 张量生成完毕！特征矩阵形状: {X_all.shape}")
    
    return X_all, y_all, dates_all

def get_dataloaders(X_all, y_all, dates_all, batch_size=1024):
    """
    严格按时间序列切分数据集，并组装零拷贝的 PyTorch DataLoader。
    """
    print("--> [1/2] 按时间边界生成切分索引 (Train/Valid/Test) ...")
    dates_series = pd.to_datetime(dates_all)

    train_indices = np.where(dates_series <= pd.to_datetime('2020-03-31'))[0]
    valid_indices = np.where((dates_series > pd.to_datetime('2020-03-31')) & (dates_series <= pd.to_datetime('2021-03-31')))[0]
    test_indices = np.where(dates_series >= pd.to_datetime('2021-03-31'))[0]

    # 抽取测试集的真实时间戳，为了后续画累计 IC 曲线单独保留
    test_dates = dates_all[test_indices].copy()

    # 销毁全局日期数组，释放内存
    del dates_series, dates_all
    gc.collect()

    print("--> [2/2] 组装全局零拷贝 TensorDataset 与 DataLoader ...")
    # torch.from_numpy 核心优势：直接使用 Numpy 数组的底层内存，不发生复制！
    full_dataset = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))

    # 只有训练集可以 Shuffle (打乱 Batch 内部顺序)
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(Subset(full_dataset, valid_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=batch_size, shuffle=False)
    
    print(f"✅ DataLoader 组装完毕！")
    print(f"   - 训练集样本数: {len(train_indices)}")
    print(f"   - 验证集样本数: {len(valid_indices)}")
    print(f"   - 测试集样本数: {len(test_indices)}")
    
    return train_loader, valid_loader, test_loader, test_dates