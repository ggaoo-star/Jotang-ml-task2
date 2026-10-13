# inference.py独立推理文件
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义模型结构（和训练时一样,见train2.py）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 检查有无显卡,没有就用CPU
model = SimpleCNN().to(device)
# 实例化刚才定义的模型,搬到对应设备上
model.load_state_dict(torch.load('cat_dog_cnn.pth', map_location=device))
# torch.load('cat_dog_cnn.pth'):从硬盘上把之前训练保存的"学习成果"(参数文件)读出来
# map_location=device 确保如果模型是在GPU上训练,现在在CPU上也能加载(反之亦然)
# model.load_state_dict(...) 把参数灌入空壳模型中

model.eval()  # 切换到评估模式
print(f"模型加载完成，使用设备: {device}")

# ==========================================
# 3. 图片预处理(和验证集一致)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. 读取本地图片并推理
img_path = r'C:\Users\16117\Desktop\Jotang-ml-task2\data\train\cat\8.jpg'  # 要预测的图片路径
image = Image.open(img_path).convert('RGB')  # 强制转成RGB三通道(模型只认)

input_tensor = preprocess(image).unsqueeze(0).to(device)
# 图片变成形状[3, 224, 224]的张量(通道,高,宽),增加一个维度:模型要求输入[batch_size, 3, 224, 224],变成[1, 3, 224, 224]
# 张量搬到和模型相同的设备

# 开始推理（不用计算梯度）
with torch.no_grad(): # 告诉不计算梯度
    output = model(input_tensor)
    # 图片给模型,得输出 output形状是[1, 2]
    probabilities = torch.nn.functional.softmax(output, dim=1)
    # Softmax:把原始分数变成概率,加起来等于1,dim=1:在类别这个维度做Softmax
    prob_cat = probabilities[0][0].item()
    prob_dog = probabilities[0][1].item()
    # probabilities[0]：取第一个(唯一个)样本概率
    # [0],[1]:分别取猫,狗的概率 .item():把PyTorch张量变成普通Python浮点数,方便打印
    _, predicted = torch.max(output, 1) # 在[1, 2]的输出里找最大值,返回最大值和所在索引
    class_names = ['cat', 'dog'] # 定义类别名,索引0对应 cat，索引1对应dog,顺序和训练时ImageFolder扫描顺序一致
    
print(f"\n预测结果: {class_names[predicted.item()]}")
print(f"猫的概率: {prob_cat:.4f}")
print(f"狗的概率: {prob_dog:.4f}")
# 打印最后结果

# 显示图片和预测结果
plt.imshow(image) # 画原图
plt.title(f"Pred: {class_names[predicted.item()]} (Cat: {prob_cat:.2f}, Dog: {prob_dog:.2f})") # 图片上方写预测结果和概率
plt.axis('off') # 关坐标轴(图片不需要)
plt.show() # 显示