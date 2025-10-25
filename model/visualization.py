import numpy as np
import matplotlib.pyplot as plt
import os

# --- 參數設置 ---
HISTORY_FILE = 'training_history.npy'

# --- 載入訓練歷史 ---
# 使用 np.load() 載入文件。
# allow_pickle=True 是必需的，因為 Keras 保存歷史記錄時使用了 Python 的字典對象。
# .item() 方法將載入的 0 維 numpy 數組轉換回 Python 字典。
print(f"正在從 '{HISTORY_FILE}' 載入訓練歷史...")
history = np.load(HISTORY_FILE, allow_pickle=True).item()
print("載入成功！")

# 打印字典中所有的鍵，方便您了解有哪些指標可以繪製
print("\n歷史記錄中包含的指標：", list(history.keys()))

# --- 繪製圖表 ---

# 設置圖表風格
plt.style.use('seaborn-v0_8-whitegrid')

# 創建一個大的圖形窗口，包含兩個子圖 (subplots)
# 2, 1 表示 2 行 1 列。figsize 是圖形的大小。
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
fig.suptitle('visulization', fontsize=16)

# --- 1. 繪製損失函數 (Loss) ---
# history['loss'] 是訓練集的損失
# history['val_loss'] 是驗證集的損失
ax1.plot(history['loss'], label='Training Loss', color='dodgerblue', linewidth=2)
ax1.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (1 - CCC)')
ax1.legend()
# 找到驗證損失最低點的 epoch
best_epoch_loss = np.argmin(history['val_loss'])
ax1.axvline(best_epoch_loss, linestyle=':', color='red', label=f'Best Epoch (Loss): {best_epoch_loss+1}')
ax1.legend() # 再次調用以顯示 axvline 的標籤


# --- 2. 繪製評估指標 (Metrics) ---
# 為了在同一個子圖中繪製兩個不同尺度的指標 (CCC 和 MAE)，我們使用兩個Y軸
# ax2 是左邊的 Y 軸，用於繪製 CCC
# ax3 是右邊的 Y 軸，用於繪製 MAE
ax2.set_title('')
ax2.set_xlabel('Epoch')

# 繪製 CCC Metric
color_ccc = 'green'
ax2.set_ylabel('CCC Metric', color=color_ccc)
ax2.plot(history['ccc_metric'], label='training CCC', color=color_ccc, alpha=0.7)
ax2.plot(history['val_ccc_metric'], label='validation CCC', color=color_ccc, linestyle='--', linewidth=2.5)
ax2.tick_params(axis='y', labelcolor=color_ccc)
# 找到驗證CCC最高的點
best_epoch_ccc = np.argmax(history['val_ccc_metric'])
ax2.axvline(best_epoch_ccc, linestyle=':', color='green', label=f'Best Epoch (CCC): {best_epoch_ccc+1}')

# 創建第二個 Y 軸，共享 X 軸
ax3 = ax2.twinx()

# 繪製 Mean Absolute Error (MAE)
color_mae = 'purple'
ax3.set_ylabel('MAE', color=color_mae)
ax3.plot(history['mean_absolute_error'], label='training MAE', color=color_mae, alpha=0.7)
ax3.plot(history['val_mean_absolute_error'], label='validation MAE', color=color_mae, linestyle='-.')
ax3.tick_params(axis='y', labelcolor=color_mae)

# 合併兩個 Y 軸的圖例
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')


# 自動調整子圖佈局，防止標籤重疊
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 調整佈局以適應主標題

# 顯示圖表
plt.show()