import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data_aug = np.load('losses_aug.npy', allow_pickle=True).item()
data_no  = np.load('losses_no_aug.npy', allow_pickle=True).item()
# np.load:硬盘加载;允许加载字典格式的数据;.npy加载回来是一个0维数组.item转回普通的python字典

# 画图
plt.figure(figsize=(12, 5))

# Loss 对比
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), data_aug['train'], 'b--', label='Train (Aug)')# b是蓝色,虚线,有增强训练集
plt.plot(range(1, 11), data_aug['val'], 'b-', label='Val (Aug)')# 蓝色,实线,有增强验证集
plt.plot(range(1, 11), data_no['train'], 'r--', label='Train (No Aug)')# 红色,虚线,无增强训练集
plt.plot(range(1, 11), data_no['val'], 'r-', label='Val (No Aug)')# 红色,实线,无增强验证集
plt.ylabel('Loss')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()
plt.grid(True) # 打开网格线

# Accuracy 对比
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), data_aug['train_acc'], 'b--', label='Train (Aug)')
plt.plot(range(1, 11), data_aug['val_acc'], 'b-', label='Val (Aug)')
plt.plot(range(1, 11), data_no['train_acc'], 'r--', label='Train (No Aug)')
plt.plot(range(1, 11), data_no['val_acc'], 'r-', label='Val (No Aug)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()