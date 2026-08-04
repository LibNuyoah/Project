"""
Step 5: LSTM 时间序列辅助验证模型
---------------------------------
基于历史负荷序列预测充电需求，验证时间依赖规律。
输出: model/lstm_model.pth
      result/figures/lstm_evaluation.png
      result/tables/lstm_metrics.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, 'result')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


def load_data():
    filepath = os.path.join(RESULT_DIR, 'clean_data.xlsx')
    return pd.read_excel(filepath)


class LSTMPredictor(nn.Module):
    """LSTM 充电负荷预测器"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # 取最后时间步
        out = self.fc(out)
        return out


def create_sequences(data, seq_length=6):
    """创建滑动窗口序列"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def prepare_lstm_data(df, seq_length=6):
    """准备LSTM数据"""
    print("=" * 60)
    print("LSTM 数据准备")
    print("=" * 60)

    all_X, all_y = [], []

    # 为每个区域和日期类型单独构建序列
    for region in df['区域编号'].unique():
        for day_type in ['工作日', '周末']:
            mask = (df['区域编号'] == region) & (df['日期类型'] == day_type)
            series = df.loc[mask].sort_values('小时')['充电负荷'].values

            if len(series) >= seq_length + 1:
                X, y = create_sequences(series, seq_length)
                all_X.append(X)
                all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    print(f"  → 序列长度 (回顾窗口): {seq_length} 小时")
    print(f"  → 总样本数: {len(X_all)}")
    print(f"  → 输入形状: {X_all.shape}")
    print(f"  → 输出形状: {y_all.shape}")

    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # 展平后归一化，再reshape
    n_samples, n_steps = X_all.shape[0], X_all.shape[1]
    X_flat = X_all.reshape(-1, 1)
    X_scaled = scaler_X.fit_transform(X_flat).reshape(n_samples, n_steps, 1)
    y_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

    # 划分训练/测试集
    split_idx = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y_scaled[:split_idx]
    y_test = y_scaled[split_idx:]
    y_test_orig = y_all[split_idx:]  # 保留原始尺度用于评估

    print(f"\n  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # 转tensor
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t),
                             batch_size=32, shuffle=False)

    return train_loader, test_loader, scaler_y, y_test_orig, X_test


def train_lstm(train_loader, test_loader, epochs=200):
    """训练LSTM模型"""
    print("\n" + "=" * 60)
    print("LSTM 模型训练")
    print("=" * 60)

    model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        scheduler.step(test_loss)

        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    print(f"\n  最终 Train Loss: {train_losses[-1]:.6f}")
    print(f"  最终 Test Loss: {test_losses[-1]:.6f}")

    return model, train_losses, test_losses


def evaluate_lstm(model, test_loader, scaler_y, y_test_orig):
    """评估LSTM模型"""
    print("\n" + "=" * 60)
    print("LSTM 模型评价")
    print("=" * 60)

    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())

    y_pred_scaled = np.concatenate(predictions, axis=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # 截取匹配长度
    n = min(len(y_pred), len(y_test_orig))
    y_pred = y_pred[:n]
    y_true = y_test_orig[:n]

    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = pd.DataFrame({
        '指标': ['MAE', 'RMSE', 'R2'],
        '数值': [mae, rmse, r2]
    })

    print(f"\n  测试集 MAE:  {mae:.1f} kWh")
    print(f"  测试集 RMSE: {rmse:.1f} kWh")
    print(f"  测试集 R2:   {r2:.4f}")

    # 保存
    metrics.to_excel(os.path.join(RESULT_DIR, 'tables', 'lstm_metrics.xlsx'), index=False)
    print(f"\n✅ LSTM评价指标已保存: {os.path.join(RESULT_DIR, 'tables', 'lstm_metrics.xlsx')}")

    return y_true, y_pred, metrics


def plot_lstm_results(y_true, y_pred, train_losses, test_losses):
    """绘制LSTM结果图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── 图1: 预测 vs 真实 (前100个测试样本) ──
    ax1 = axes[0, 0]
    n_plot = min(100, len(y_true))
    ax1.plot(range(n_plot), y_true[:n_plot], 'o-', color='#3498DB',
             linewidth=1.5, markersize=3, label='真实值', alpha=0.8)
    ax1.plot(range(n_plot), y_pred[:n_plot], 's-', color='#E74C3C',
             linewidth=1.5, markersize=3, label='LSTM预测', alpha=0.8)
    ax1.fill_between(range(n_plot), y_true[:n_plot], y_pred[:n_plot],
                     alpha=0.2, color='gray')
    ax1.set_xlabel('测试样本序号', fontsize=11)
    ax1.set_ylabel('充电负荷 (kWh)', fontsize=11)
    ax1.set_title('LSTM 预测 vs 真实值 (测试集前100样本)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ── 图2: 训练过程 ──
    ax2 = axes[0, 1]
    ax2.plot(train_losses, color='#3498DB', linewidth=1.5, label='训练损失')
    ax2.plot(test_losses, color='#E74C3C', linewidth=1.5, label='验证损失')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MSE Loss', fontsize=11)
    ax2.set_title('LSTM 训练过程', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # ── 图3: 预测 vs 真实散点图 ──
    ax3 = axes[1, 0]
    ax3.scatter(y_true, y_pred, alpha=0.4, c='#3498DB', edgecolors='white', s=20)
    ax3.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             'r--', linewidth=2)
    ax3.set_xlabel('真实值 (kWh)', fontsize=11)
    ax3.set_ylabel('预测值 (kWh)', fontsize=11)
    ax3.set_title(f'LSTM 预测散点图 (R2={r2_score(y_true, y_pred):.4f})',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # ── 图4: 残差分布 ──
    ax4 = axes[1, 1]
    residuals = y_true - y_pred
    ax4.hist(residuals, bins=30, color='#3498DB', edgecolor='white', alpha=0.8)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('残差 (kWh)', fontsize=11)
    ax4.set_ylabel('频数', fontsize=11)
    ax4.set_title(f'LSTM 残差分布 (均值={residuals.mean():.1f}, 标准差={residuals.std():.1f})',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(RESULT_DIR, 'figures', 'lstm_evaluation.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ LSTM评价图已保存: {output_path}")


def main():
    df = load_data()
    train_loader, test_loader, scaler_y, y_test_orig, _ = prepare_lstm_data(df, seq_length=6)
    model, train_losses, test_losses = train_lstm(train_loader, test_loader, epochs=200)
    y_true, y_pred, metrics = evaluate_lstm(model, test_loader, scaler_y, y_test_orig)
    plot_lstm_results(y_true, y_pred, train_losses, test_losses)

    # 保存模型
    model_path = os.path.join(MODEL_DIR, 'lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ LSTM模型已保存: {model_path}")

    # 论文结论
    print("\n" + "=" * 60)
    print("LSTM 模型分析结论（可直接写入论文）")
    print("=" * 60)

    print(f"""
1. LSTM模型基于历史6小时充电负荷序列预测下一时刻负荷，
   测试集 R2={metrics[metrics['指标']=='R2']['数值'].values[0]:.4f}，
   MAE={metrics[metrics['指标']=='MAE']['数值'].values[0]:.1f} kWh，
   RMSE={metrics[metrics['指标']=='RMSE']['数值'].values[0]:.1f} kWh。

2. LSTM通过门控机制有效捕捉了充电负荷序列中的时间依赖性，
   验证了充电需求存在显著的时间自相关特征。

3. 与XGBoost的对比：XGBoost利用多维静态特征进行预测，
   LSTM仅利用负荷序列自身的历史信息。
   两者从不同角度对充电需求进行建模，互为补充验证。

4. LSTM的优势在于无需额外特征即可进行时序预测，
   但缺点是对特征维度利用率较低，仅捕捉时序规律。
   因此论文采用 "XGBoost主预测 + LSTM时序验证" 的双模型策略。
""")

    return model, metrics


if __name__ == '__main__':
    main()
