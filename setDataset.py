import cv2
import torch, torchvision
import glob
import os, time, random
import numpy
import csv
import visdom
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 连连看图片块数据集生成
class LinkBlocks(Dataset):
    def __init__(self, root, resize, mode):
        super(LinkBlocks, self).__init__()
        # 保存参数
        self.root = root
        self.resize = resize
        # 给每一个类做映射
        self.name2label = {} # "背景"：0；"寿司"：1……
        # os.listdir每次返回目录下的文件列表顺序会不一致，
        # 排序是为了每次返回文件列表顺序一致
        for name in sorted(os.listdir(os.path.join(root))):
            # 过滤掉非目录文件
            if not os.path.isdir(os.path.join(root, name)):
                continue
            # 一个文件夹中的图片的label值均相同
            self.name2label[name] = len(self.name2label.keys())
        # 输出label值
        # print(self.name2label)
        # 加载文件
        self.images, self.labels = self.load_csv('blocks.csv')
        # 剪裁数据
        # if mode == 'train':
        #     self.images = self.images[:int(0.8 * len(self.images))]  # 将数据集的60%设置为训练数据集合
        #     self.labels = self.labels[:int(0.8 * len(self.labels))]  # label的60%分配给训练数据集合
        # elif mode == 'test':
        #     self.images = self.images[int(0.8 * len(self.images)):int(1.0 * len(self.images))]  # 从60%-80%的地方
        #     self.labels = self.labels[int(0.8 * len(self.labels)):int(1.0 * len(self.labels))]
        # else:
        #     self.images = self.images[int(0.8 * len(self.images)):]  # 从80%的地方到最末尾
        #     self.labels = self.labels[int(0.8 * len(self.labels)):]

    # 将目录下的图片路径与其对应的label写入csv文件，并将csv文件中内容读出，返回为图片名和label
    def load_csv(self, filename):
        # 如果不存在csv文件则先创建
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            # 获取指定目录下所有的满足后缀的图像名
            # 如：DataSet/00/00_00_01.jpg
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            # 如：256 ‘DataSet/00/00_00_01.jpg’
            print(len(images), images)
            # 将images列表中的数据随机排序
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'DataSet/00/00_00_01.jpg'
                    cls = img.split(os.sep)[-2]
                    label = self.name2label[cls]
                    writer.writerow([img, label])
                print("write into csv into :", filename)

        # 如果存在csv文件则读取csv文件
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 接下来就会得到 'DataSet/00/00_00_01.jpg' 0 的对象
                img, label = row
                images.append(img)
                labels.append(int(label))

        # 保证images和labels的长度是一致的
        assert len(images) == len(labels)
        return images, labels

    # 返回数据的数量
    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean
        return x

    # 返回x的数据和当前图片的label
    def __getitem__(self, x):
        # idex-[0-总长度]
        # retrun images,labels
        # 将图片，label的路径取出来
        # 得到的img是这样的一个类型：'DataSet/00/00_00_01.jpg'
        # 然而label得到的则是 0，1，2 这样的整形的格式
        img, label = self.images[x], self.labels[x]
        img_process = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # 将t图片的路径转换可以处理图片数据
            transforms.Resize((self.resize, self.resize)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
            # # 随机旋转
            # transforms.RandomRotation(15),  # 设置旋转的度数小一些，否则的话会增加网络的学习难度
            # # 中心裁剪
            # transforms.CenterCrop(self.resize), # 此时：既旋转了又不至于导致图片变得比较的复杂
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = img_process(img)
        label = torch.tensor(label)
        return img, label

# 图片重命名
def rename(path):
    dirlist = os.listdir(path)  # 获取文件路径
    dir_num = len(dirlist)  # 获取目录长度（个数）
    # print(dirlist)
    for dirname in dirlist:
        if not os.path.isdir(os.path.join(path, dirname)):
            continue
        imglist = os.listdir(os.path.join(path, dirname))
        for imgname in imglist:
            if imgname.endswith('.jpg'):  # 文件名以.jpg结尾的
                src = os.path.join(os.path.abspath(path), dirname, imgname)
                print(src)
                dst = os.path.join(os.path.abspath(path), dirname + ".jpg")  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                print("dst=",dst)
                try:
                    os.rename(src, dst)
                    # print('converting %s to %s ...' % (src, dst))
                except:
                    continue
    print('total %d to rename & converted jpgs' % (dir_num))

# 循环移位
def cyc_trans(img, ROW = 101, COL = 102):
    """遍历图像每个像素的每个通道"""
    # print(img.shape)  # 打印图像的高，宽，通道数
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    left = numpy.zeros([h, w, c], dtype=numpy.uint8)
    right = numpy.zeros([h, w, c], dtype=numpy.uint8)
    up = numpy.zeros([h, w, c], dtype=numpy.uint8)
    down = numpy.zeros([h, w, c], dtype=numpy.uint8)
    for row in range(h):  # 遍历每一行
        for col in range(w):  # 遍历每一列
            for channel in range(c):  # 遍历每个通道（三个通道分别是BGR）
                down[(row+10)%ROW][(col)%COL][channel] = img[row][col][channel]
                right[(row)%ROW][(col+10)%COL][channel] = img[row][col][channel]
                up[(row-10)%ROW][(col)%COL][channel] = img[row][col][channel]
                left[(row)%ROW][(col-10)%COL][channel] = img[row][col][channel]
    return up, down, left, right

# 生成上下左右循环平移图片
def img_create(path):
    dirlist = os.listdir(path)  # 获取文件路径
    dir_num = len(dirlist)  # 获取目录长度（个数）
    print(dirlist)
    for dirname in dirlist:
        if not os.path.isdir(os.path.join(path, dirname)):
            continue
        imglist = os.listdir(os.path.join(path, dirname))
        for imgname in imglist:
            if imgname.endswith('00.jpg'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                src = os.path.join(os.path.abspath(path), dirname, imgname)
                print("src=", src)
                img = cv2.imread(src)
                obj, cls, num = get_image_name(src)

                up, down, left, right = cyc_trans(img, 101, 102)
                cv2.imwrite(".\\DataSet\\" + dirname + "\\" + str(obj) + "_" + str(cls) + "_01.jpg", up)
                cv2.imwrite(".\\DataSet\\" + dirname + "\\" + str(obj) + "_" + str(cls) + "_02.jpg", down)
                cv2.imwrite(".\\DataSet\\" + dirname + "\\" + str(obj) + "_" + str(cls) + "_03.jpg", left)
                cv2.imwrite(".\\DataSet\\" + dirname + "\\" + str(obj) + "_" + str(cls) + "_04.jpg", right)

                print("obj {}, cls {}, num {}".format(obj,cls,num))

# 获取图片名
def get_image_name(image_name):
    image_name = os.path.split(image_name)[-1]
    image_name = image_name.split('.')[0]
    obj, cls, num = image_name.split('_')
    return obj, cls, num

if __name__ == "__main__":
    # 图片绝对路径
    path = "F:\\Program\\Python\\AutoLink\\AutoLinkGames-master\\Auxiliary\\DataSet"
    # 图片重命名为统一格式
    # rename(path)

    # 图片平移扩充数据集
    # img_create(path)

    # 验证, 需要在Terminal中输入 python -m visdom.server
    viz = visdom.Visdom()

    # 创建数据集
    trainset = LinkBlocks('.\\Auxiliary\\DataSet', 32, 'train')
    trainloader = DataLoader(trainset, batch_size=5, shuffle=True, num_workers=2)
    index = 1
    for image, label in trainloader:
        print(index)
        index+=1
        viz.images(trainset.denormalize(image), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(label.numpy()), win='label', opts=dict(title='batch-y'))

        # time.sleep(1)

    testset = LinkBlocks('.\\Auxiliary\\DataSet', 32, 'test')
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    index = 1
    for image, label in testloader:
        print(index)
        index += 1
        viz.images(testset.denormalize(image), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(label.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(1)