import torchvision
from torch.utils.data import DataLoader
import numpy as np
import imageio
 
#train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
#test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
 
 
# 解压文件函数 返回解压后的字典
def unpickle(file):
    import pickle as pk
    fo = open(file, 'rb')
    dict = pk.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict
 
 

root_dir="./dataset/"
# 生成训练集图片
for j in range(1, 6):
    dataName = root_dir+"/data_batch_" + str(j)  # 读取当前目录下的data_batch1~5文件。
    Xtr = unpickle(dataName)
 
    for i in range(0, 9600):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        picName = root_dir+'/train/' + str(Xtr['labels'][i]) + '_' + str(i + (j - 1) * 9600) + '.jpg'
        imageio.imsave(picName, img)  # 使用的imageio的imsave类
 
# 生成验证集图片
    for i in range(9600, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        picName = root_dir+'/validation/' + str(Xtr['labels'][i]) + '_' + str(i + (j - 1) * 480) + '.jpg'
        imageio.imsave(picName, img)  # 使用的imageio的imsave类

# 生成测试集图片
testXtr = unpickle(root_dir+"/test_batch")
for i in range(0, 6000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = root_dir+'/test/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
    imageio.imsave(picName, img)

# 生成验证集图片
for i in range(6000, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = root_dir+'/validation/' + str(testXtr['labels'][i]) + '_' + str(i-6000+2000) + '.jpg'
    imageio.imsave(picName, img)