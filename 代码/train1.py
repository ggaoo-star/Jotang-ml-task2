# ===========2读取和转换=============
import numpy as np
from PIL import Image # 图像处理库
import torch
import matplotlib.pyplot as plt #画图工具

# 1 读取图片
img_pil = Image.open('highway.jpg')

# 2 转换为NumPy数组,变成三维数学矩阵方便数学计算 HWC格式(高,宽,通道数)
# 如一张800x600的彩色图，形状就是(800,600,3) 3代表红绿蓝三个通道
img_np = np.array(img_pil)

# 3 NumPy数组转换为 PyTorch Tensor (CHW 格式)
# torch.from_numpy得到的还是HWC，需要permute换轴,换为CHW(通道,高,宽)
# 是PyTorch的硬性规定 卷积层要求输入的通道数在最前面
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

# 4 打印信息 
print("=== NumPy 数组信息 (HWC) ===")
print(f"Shape: {img_np.shape}") # 输出 (高, 宽, 3)
print(f"Dtype: {img_np.dtype}") # 数据类型 通常照片是uint8(无符号8位整数),即数字范围在0到255间
print(f"Max: {img_np.max()}, Min: {img_np.min()}") # 打印整张图里最大和最小的像素值

print("\n=== PyTorch Tensor 信息 (CHW) ===")
print(f"Shape: {img_tensor.shape}") # 输出 (3, 高, 宽)
print(f"Dtype: {img_tensor.dtype}") # 保留原本的数据类型
# 但在真正送入神经网络前,我们还会把它转成float32并做归一化(按比例缩放到统一范围)消除量纲(物理量大小和单位有关)、提高模型训练稳定性并加速收敛

# 5 打印一个像素的RGB值(取左上角第一个像素) 即0行0列
pixel_rgb = img_np[0, 0, :]
print(f"\n左上角第一个像素的 RGB 值: {pixel_rgb}")

#========手写二维卷积函数=========
def my_conv2d(image, kernel, padding=0, stride=1):
# 定义函数的关键字;函数名;image:输入参数,代表一张二维图像,是NumPy矩阵
# kernel:输入参数,代表卷积核(提取输入数据的局部特征),一个小一点的二维矩阵
# padding=0:vaild 四周补上0圈像素(默认),边缘直接参与计算,卷积核在边缘会“悬空”一部分(卷积核只能放在能完整盖住图片的位置,输出图比输入图小)
# stride=1:步长 卷积核每次滑动几格

    # 1 获取矩阵的尺寸
    H, W = image.shape # 返回图像的高和宽负值给H和W
    kH, kW = kernel.shape # 返回卷积核的高和宽赋值给KH和KW

    # 2 处理图像边缘Padding(填充)
    # 如果是valid(padding=0),则不填充；如果是same,则要计算
    if padding > 0:
        # 在图像四周填充0
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        # np.pad:NumPy的填充函数;括号中表示在高度和宽度方向各填padding列;mode='constant':填充方式是“用常数填充" constant_values=0 常数的值是0(即黑边）
        # why黑边:灰度图中0表示最暗即黑色 彩色图0也是最暗,即(0,0,0)为黑色,所以补0为黑边
    else:
        padded_image = image #直接用原图

        #更新填充后的尺寸
    pH, pW = padded_image.shape
    # 获取填充后图像的新高度pH和宽度pW,没填充就不变(H,W)

    # 3 计算输出特征图尺寸
    # 公式:(输入尺寸+2*Padding-卷积核尺寸)/步长+1
    out_H = (pH - kH) // stride + 1
    out_W = (pW - kW) // stride + 1

    # 4 初始化输出矩阵
    output = np.zeros((out_H, out_W))
    # 先建一个全为0的矩阵,大小是刚才算出来的out_H和out_W

    # 5 开始滑动窗口(核心循环)
    for i in range(out_H): # i是行索引
        for j in range(out_W): # j是列索引
    # 两个循环遍历了输出矩阵的每个位置

            # 找到当前窗口在原图上的起始和结束位置
            h_start = i*stride # 第i个输出位置对应原图的行起始位置是i*stride
            h_end = h_start+kH # 结束位置是起始位置加上卷积核的高度
            w_start = j*stride # 第j个输出位置对应原图的列起始位置是ji*stride
            w_end = w_start+kW # 结束位置是起始位置加上卷积核的宽度

            # 截取当前窗口区域
            region = padded_image[h_start:h_end, w_start:w_end]
            # 利用NumPy的切片功能,将原图上h_start-h_end,w_start-w_end这个矩形区域扣出,存在region变量里,region的尺寸和卷积核一样

            # 对应位置相乘再求和 (点积)
            output[i, j] = np.sum(region * kernel)
            # 卷积的数学本质:抠出来的图像区域和卷积核对应位置相乘(不是矩阵乘法,是元素级相乘)
            # np.sum(...):将相乘得到的所有数字加起来,将算出的这个总和填入输出矩阵(i,j)的位置
            # 完成一次卷积运算,循环继续

    return output

#==========四种卷积核处理==========
# 把彩色图转为灰度图方便处理(手写函数目前只支持单通道,卷积核通常是一个二维矩阵)
gray_img = np.array(img_pil.convert('L'))
# my_conv2d函数只能处理二维矩阵,彩色图片img_pil，用.convert('L')转成灰度图(L为亮度),用np.array()转成NumPy矩阵

#1.均值模糊(取周围3*3区域所有像素的平均值,抹平细节)
mean_kernel = np.ones((3, 3)) / 9.0
# 生成一个3x3的全1矩阵,除以9让矩阵元素和为1
blur_mean = my_conv2d(gray_img, mean_kernel, padding=1) 
# padding=1 保证输出尺寸和原图一样

# 2.高斯模糊(中心像素更高的权重，给边缘像素更低的权重,模糊出来图片更自然)
# 计算好的5x5高斯核 (sigma=1.0) 用NumPy自动生成
# def get_gaussian_kernel(size=5, sigma=1.0):
    #ax = np.arange(-size // 2 + 1., size // 2 + 1.)  -生成坐标轴
    #xx, yy = np.meshgrid(ax, ax)  -生成网络坐标
    #kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))  -套用二维高斯公式
    #return kernel / np.sum(kernel)  # 归一化
# gaussian_kernel = get_gaussian_kernel(size=5, sigma=1.0)
# print(gaussian_kernel)
gaussian_kernel = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]) / 256.0 # 保证权重之和为1,不会让图片整体变亮或变暗(权重归一化)
blur_gaussian = my_conv2d(gray_img, gaussian_kernel, padding=2) 
# padding=2 保证尺寸不变

# 锐化(放大差异)
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened = my_conv2d(gray_img, sharpen_kernel, padding=1)
# 中心是5(正数,放大中心像素),上下左右是-1(负数,减去周围像素)
# 计算:中心像素*5-上-下-左-右 算出来越大越亮

# 边缘检测
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
# 检测垂直边缘:计算水平方向的差值(右-左) 如果此位置有垂直边缘(左黑右白),结果大,显示为亮线;结果接近0:不亮
sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
# 检测水平边缘:计算垂直方向差值:下-上 有水平边缘(上黑下白),结果大,亮线
# 中间的数字2和-2是加权，让靠近中心的像素影响更大(Sobel算子认为中心区域更重要)
edge_x = my_conv2d(gray_img, sobel_x, padding=1)
# 用Sobel-X卷积,得到的结果,垂直边缘被高亮显示,其他区域接近黑色
edge_y = my_conv2d(gray_img, sobel_y, padding=1)
# 用Sobel-Y卷积,得到的结果里,水平边缘被高亮显示
edge_combined = np.sqrt(edge_x**2 + edge_y**2)
# 勾股定理:由x方向变化量和y方向变化量求总变化量(梯度幅值),把两个方向的边缘信息合并成了一张完整的边缘图
# 也可以用np.abs(edge_x) + np.abs(edge_y)来合并

#=====测试Padding和Stride对输出尺寸的影响=====
import numpy as np
from PIL import Image
# 最前面已经导入过
# img_pil是读出来的图片Tensor，形状为(H,W,C)

# 1 准备灰度图（前面的my_conv2d只接受二维矩阵）
img_pil_gray = Image.open('highway.jpg').convert('L') 
# 重新读取并转为灰度，避免受前面彩色变量的干扰
img_np_gray = np.array(img_pil_gray, dtype=np.float32) 
# 把PIL的图片对象转换成NumPy数组(矩阵) 变成32位浮点数方便计算(原图是uint8 只能存0-255的整数)
print(f"\n灰度图形状: {img_np_gray.shape}, 类型: {img_np_gray.dtype}")

# 2. 定义一个简单的均值卷积核(3*3)
test_kernel = np.ones((3, 3)) / 9.0

# 3. 运行Valid模式(不填充)
out_valid = my_conv2d(img_np_gray, test_kernel, padding=0, stride=1)
# 调用手写的卷积函数
print(f"Valid 模式输出形状: {out_valid.shape}")  
# 输出尺寸会变小 卷积核无法覆盖到最边缘的像素，所以丢失了边缘信息

# 4. 运行Same模式(填充)
out_same = my_conv2d(img_np_gray, test_kernel, padding=1, stride=1)
print(f"Same 模式输出形状: {out_same.shape}")  
# 输出尺寸和原图一样 因为补了边

# 5. 运行stride=2模式(卷积核每次滑动2格)
out_stride2 = my_conv2d(img_np_gray, test_kernel, padding=1, stride=2)
print(f"Stride=2 输出形状: {out_stride2.shape}") 
# 输出尺寸减半

#====图像卷积完成后，部分结果可能小于0或大于原来的像素范围:====
# 截断:小于0的全部变成0,大于255的全部变成255-np.clip
clipped = np.clip(edge_combined, 0, 255) 
# edge_combined 是算出来的包含负数的矩阵)

# 类型转换:之前计算用的float32 照片格式只接受unit8,会直接砍掉小数部分
final_img = clipped.astype(np.uint8)

# 转回PIL对象(数组变回图片对象)
pil_image = Image.fromarray(final_img)

# 保存 
pil_image.save('result.jpg') 
print("已保存: result.jpg")


