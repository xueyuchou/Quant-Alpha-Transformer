import time
import torch
import torch.nn as nn
import torch.optim as optim
import random

from utils.data_pipeline import load_and_preprocess_data, create_time_series_tensors, get_dataloaders
from utils.metrics import evaluate_model, compute_and_plot_ic
from models.rope_transformer import QuantTransformer

def set_seed(seed=4396):
    """固定全局随机数种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    set_seed(4396)
    # ==========================================
    # 1. 数据准备阶段 (调用 utils/data_pipeline.py)
    # ==========================================
    # 注意：这里你可以改成 toy_data.csv 进行快速测试，或者用真实的 parquet 测试
    parquet_file = 'data/toy_data.csv'  
    df_full, feature_cols = load_and_preprocess_data(parquet_file)
    X_all, y_all, dates_all = create_time_series_tensors(df_full, feature_cols, seq_len=20)
    train_loader, valid_loader, test_loader, test_dates = get_dataloaders(X_all, y_all, dates_all, batch_size=1024)

    # ==========================================
    # 2. 训练准备阶段 
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n 当前计算引擎: {device.type.upper()}")

    num_features = 158
    model = QuantTransformer(embed_size=num_features, output_size=1, seq_len=20).to(device)
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print("开始训练 (RoPE + Attention + SmoothL1 + 早停)...")
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_path = 'best_attention_model.pth'
    num_epochs = 15

    # ==========================================
    # 3. 核心训练循环
    # ==========================================
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()

            # 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                val_loss += criterion(model(batch_X.to(device)), batch_y.to(device)).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f} | 耗时: {time.time()-start_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(" 锁定最优状态！")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" 连续 {patience} 轮未提升，早停机制减少过拟合！")
                break

    # ==========================================
    # 4. 加载最优权重，执行样本外回测评估
    # ==========================================
    print("\n加载最优未过拟合模型进行评估...")
    model.load_state_dict(torch.load(best_model_path))
    
    # 运行推理并获取结果
    predictions, y_trues = evaluate_model(model, test_loader, device)
    
    # 计算 IC 并画图
    compute_and_plot_ic(predictions, y_trues, test_dates)

if __name__ == "__main__":
    main()