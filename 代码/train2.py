#==================整理数据===================
import os                          # 导入Python自带的os(操作系统),用来处理文件路径
import torch                       # PyTorch核心库
import torch.nn as nn              # PyTorch的神经网络工具箱(包含卷积层、全连接层等)
import torch.optim as optim        # PyTorch的优化器工具箱(如Adam,SGD更新神经网络的参数）
from torch.utils.data import DataLoader, random_split 
# 从PyTorch的工具箱里，拿出DataLoader(把图片打包成Batch)和random_split(把数据集切成训练集和验证集)
from torchvision import datasets, transforms           
# 把视觉工具库(torchvision)里的datasets(自动读取文件夹里的图片),transforms(对图片做预处理如缩放,旋转)拿出来
import matplotlib.pyplot as plt    # 导入画图工具(后面画Loss/Acc曲线)
import numpy as np                 # 矩阵计算库

#==================数据预处理====================
train_transform = transforms.Compose([
# 定义一个"训练集流水线",用compose把步骤串起来
        transforms.Resize((224, 224)),
# 神经网络要求输入尺寸必须一致,所以把所有图片的尺寸改成224像素宽,224像素高(原图尺寸不一)
            transforms.RandomHorizontalFlip(),
# 随机水平翻转(50%) 使数据增强(比如猫朝左,翻转后朝右,模型能学到"不管朝哪都是猫")
                transforms.RandomRotation(10),
# 随即旋转,角度再-10和10之间 使数据增强(模拟拍摄时相机倾斜情况)
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
# 随机调整亮度和对比度,幅度是2.0 数据增强(模拟不同光照条件下的照片)
                        transforms.ToTensor(),
# 图片转换为PyThon里面的Tensor(张量)图片形状HWC-CWH 像素值0-255变成0.0-1.0的浮点数
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 标准化:对每个通道,减去均值,再除以标准差;让数据分布更接近标准正态分布,加速模型训练收敛;这组数字是ImageNet数据集(大规模标注图像数据库)统计出来的“黄金标准值”
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 统一尺寸
    transforms.ToTensor(), # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化(和训练集一致)
])
# 定义验证集流水线;验证集不能做数据增强(翻转,旋转,调颜色),只做必要的预处理(Resize,ToTensor,Normalize)
# 验证集做客观评价,不可改变原图

#===========加载数据集 划分==============
data_dir = './data/train'
# 当前目录下-data 文件夹-train 文件夹
# ======= 自动检测并删除损坏图片 =======
import os
from PIL import Image

data_dir = './data/train'
print("正在检查数据集中是否有损坏图片...")
bad_count = 0
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify() 
            except Exception as e:
                print(f"发现坏图，已删除: {file_path}")
                os.remove(file_path)
                bad_count += 1
print(f"检查完成，共删除 {bad_count} 张损坏图片。\n")

# 然后再加载数据集
# 图片所在路径是当前目录下的data/train文件夹
full_dataset = datasets.ImageFolder(root=data_dir)
# 用ImageFolder扫描此文件夹,自动识别子文件夹(cat和dog),每个类别分配一个数字标签,并把所有图片路径和标签存起来
print(f"数据集类别: {full_dataset.classes}")
print(f"数据集总数量: {len(full_dataset)}")

# 打印类别列表和总数
train_size = int(0.8 * len(full_dataset))
# 训练集大小=总数据*80% int取整,防止出现小数张照片
val_size = len(full_dataset) - train_size
# 验证集大小=总数-训练集大小(即20%)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# 把full_dataset随机切分成两份,长度是train_size和val_size
train_dataset.dataset.transform = train_transform
# 把train_transform流水线绑定到训练集上,random_split切分后的子集默认没有transform属性
# 必须有这行数据增强才生效
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 创建一个训练集数据加载器 每次打包32张图一起送进模型,每个Epoch打乱顺序
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# 创建验证集数据加载器,验证集不需要打乱
images, labels = next(iter(train_loader))
# 检查数据格式是否正确:iter(train_loader)把加载器变成迭代器(遍历容器元素),next()从中取出第一个Batch的数据
print(f"\n一个 Batch 的图像形状: {images.shape}")
print(f"一个 Batch 的标签形状: {labels.shape}")
# 打印形状(应该是(32,3,224,224)即32张图,3通道,224*224)标签为32


#==============搭建CNN模型=================
class SimpleCNN(nn.Module):
# 定义一个类SimpleCNN,继承自nn.Module,PyTorch规矩:所有神经网络必须继承nn.Module
        def __init__(self):
            super(SimpleCNN, self).__init__()
# 构造函数(初始化) super()._init_()是固定写法,先初始化父类
            self.features = nn.Sequential(
# 定义一个叫features(特征提取器)的容器,用Sequential把后面的层按顺序串起来
                nn.Conv2d(3, 16, kernel_size=3, padding=1),  
# 卷积层为1,输入通道=3,输出通道=16,卷积核大小3*3,Padding=1
                nn.ReLU(),                  
# 激活函数ReLU,将负数变为0,正数不变  引入非线性,让网络能拟合复杂函数                  
                nn.MaxPool2d(kernel_size=2, stride=2),        
# 池化层1(缩减特征图尺寸,提取核心特征并提高计算效率) 用2x2的窗口,步长2,取最大值  使宽高减半,保留最显著的特征
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  
# 卷积层2 + ReLU + 池化层 2      16-32 通道,尺寸 112-56   
                nn.Conv2d(32, 64, kernel_size=3, padding=1), 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),   
# 卷积层3 + ReLU + 池化层 3      32-64 通道,尺寸 56-28
        )
            self.classifier = nn.Sequential(
            nn.Flatten(),
# 定义分类器(classifier)容器,nn.Flatten()：把(64,28,28)的特征图“拍扁”成64*28*28=50176 的一维向量
            nn.Linear(64 * 28 * 28, 128),
# 全连接层1 输入50176维,输出128维
            nn.ReLU(),
# 激活函数ReLU
            nn.Linear(128, 2)
# 全连接层2 输入128维,输出2维(对应猫和狗两个类别)
        )
        def forward(self, x):  # 前向传播函数 x是输入图片
            x = self.features(x)  # x先经过特征提取器(3个卷积块)
            x = self.classifier(x) # x再经过分类器(全连接层)
            return x        # 返回最终输出(2个数值,猫和狗的分数)
# 1. 定义设备（有显卡用显卡，没显卡用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. 实例化模型，并把它搬到 device 上
model = SimpleCNN().to(device)
# 3. 打印一下当前用的是什么设备（方便确认）
print(f"当前使用的设备是: {device}")
print("\nCNN 模型结构:")
print(model)  # 打印模型结构,检查每一层是否正确
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失,衡量模型输出与真实标签的差距
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器Adam，学习率lr=0.001 根据损失,自动调整模型里的所有参数 


#==============训练循环===============
num_epochs = 10  # 训练10轮
train_losses, val_losses = [], [] # 用来装每轮的 Loss，方便后面画图
train_accs, val_accs = [], [] # 用来装每轮的 Accuracy

for epoch in range(num_epochs): # 大循环
    model.train()                      # 到训练模式,启用Dropout,BatchNorm等训练时才用的层
    running_loss = 0.0                 # 累加这一轮损失
    correct_train = 0                  # 记录预测对了多少张
    total_train = 0                    # 记录一共看了多少张

    for images, labels in train_loader: # train_loader每次吐出32张图和对应的标签,直到训练集所有图片都过一遍
        images, labels = images.to(device), labels.to(device)
        #把数据搬到 GPU/CPU 上，保证模型和数据在同一块设备上
        
        optimizer.zero_grad()        # 清空旧梯度   
        outputs = model(images)         # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()    # 反向传播             
        optimizer.step()     # 更新参数           
        
        running_loss += loss.item() # 这一批的损失值(张量变成普通数字)加到总损失里
        _, predicted = torch.max(outputs.data, 1)  #outputs是32行2列的数据,每一行里找最大值,返回最大值和最大值所在的索引
        total_train += labels.size(0) 
        correct_train += (predicted == labels).sum().item()
        # labels.size(0)是这一批的样本数(32) (predicted == labels) 逐个对比预测和真实标签，相等返回1,不相等返回0，.sum()加起来是这一批预测正确的数量
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    # 循环结束，总损失除以批次数得平均损失,正确数除以总数得准确率 存进账本

    model.eval()     # 进入评估模式                 
    running_val_loss = 0.0 #上下文管理器(不用计算梯度,验证阶段不用更新参数，省显存和计算时间)
    correct_val = 0
    total_val = 0

    with torch.no_grad():      # 上下文管理器,验证时不计算梯度，节省显存
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item() #和训练阶段几乎一样,只是少三步(不算梯度)

    val_loss = running_val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accs.append(val_acc) # 计算验证集的平均损失和准确率，存进“账本”

    # 打印本轮结果
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print("训练完成！")


# =============保存模型==============
torch.save(model.state_dict(), 'cat_dog_cnn.pth')
# model.state_dict() 是模型学到的所有参数(权重和偏置),torch.save 存成一个文件
print("模型已保存为 cat_dog_cnn.pth")


# ===========特征图可视化============
model.eval() # 切换到评估模式
with torch.no_grad(): # 关闭梯度计算
        sample_images, sample_labels = next(iter(train_loader))
        # train_loader里随机拿一个Batch的数据,DataLoader变成迭代器,迭代器里取出一组数据(32张图+32个标签)
        # sample_images:[32, 3, 224, 224] sample_labels:[32]
        sample_img = sample_images[0:1].to(device)
        # 32张里取第一张搬到GPU上,[0:1]保留batch维度(模型要求),张量搬到和模型相同的设备
        first_conv_output = model.features[0](sample_img)
        # model.features:特征提取容器[0]:第一层卷积,图片给这层卷积,得输出first_conv_output形状
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        # 创建画布:4行4列,fig:画布,axes:16个格子组成数组,axes.flat可把他拉成一维,方便循环
fig.suptitle('First Conv Layer Feature Maps (16 Channels)', fontsize=16)
        # 给画布加一个大标题
for i, ax in enumerate(axes.flat): 
        # 遍历16个格子,拉平成16个元素的数组,enumerate同时获取索引i(0-15)和格子对象ax
    feature_map = first_conv_output[0, i].cpu().numpy()
    # 取出第i个通道的特征图,张量从GPU搬到CPU(matplotlib只画),PyTorch张量转成numpy数组(matplotlib画)
    ax.imshow(feature_map, cmap='viridis')
    # 前格子里画这张特征图,cmap='viridis':颜色映射 深紫色=低激活(弱响应),亮黄=高激活(强)'gray':黑白图
    ax.set_title(f'Channel {i}')
    # 每个子图加标题标明是第几个通道
    ax.axis('off')
    # 关掉子图的坐标轴
plt.tight_layout()
# 调整间距,使美观
plt.show()
#显示画布


# ============绘制曲线===============
plt.figure(figsize=(12, 4))
# 准备画布 宽12(英寸) 高4(英寸)
# 1. Loss 曲线
plt.subplot(1, 2, 1) # 把画布切成格子:1行2列(两个格子),现在再第一个(左边的)格子画Loss曲线
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
# 画折线图 range(1, num_epochs+1):横坐标(x),代表第一轮到第十轮(num_epochs=10),train_losses:纵坐标(Y),装训练集每轮Loss值,这条线起名为“Train Loss”
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
# 同上一行,画的是验证集的Loss曲线
plt.xlabel('Epoch') #给x轴贴标签:Epoch(轮次)
plt.ylabel('Loss') # 给y轴贴标签:Loss(损失值)
plt.title('Loss Curve') # 这张图起大标题:Loss曲线
plt.legend() # 显示图例:把label起的名字(Train Loss和Val Loss)显示在角落,显示哪条是哪条
plt.grid(True) # 打开网格线,画格子方便看数值

# 2. Accuracy 曲线
plt.subplot(1, 2, 2) #依旧是1行2列,但切换到第二个格子
plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True) 
# 同上面代码一样逻辑,横坐标依旧为Epoch(轮次),纵坐标切换为准确率Accuracy
# 画了两条线:训练集准确率(train_acc)和验证集准确率(val_acc),最后加上坐标轴名字,标题,图例,网格

plt.tight_layout() # 自动调整间距,排版更美观
plt.show() # 画图


# 保存数据,方便后面画对比图
import numpy as np
np.save('losses_aug.npy', {'train': train_losses, 'val': val_losses, 'train_acc': train_accs, 'val_acc': val_accs})
print("有增强数据已保存！")