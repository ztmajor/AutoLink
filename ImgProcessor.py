import cv2
import numpy
import os
import time
from alexnet import Net
import OutputSimulator as outs
from PIL import Image
import torch
import torchvision.transforms as transforms

ROW_NUM = 7
COL_NUM = 6
global grid_height, grid_width

def createMatrix(game_area):
    global  grid_height, grid_width
    num_matrix = {}

    grid_height = int(game_area.shape[0] / ROW_NUM)
    grid_width = int(game_area.shape[1] / COL_NUM)

    device = torch.device('cuda')
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mynet = torch.load('alexnet.pth')
    mynet = mynet.to(device)
    mynet.eval()
    torch.no_grad()

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
               10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
               20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
               30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
               40, 41)

    num_matrix[0] = {}
    for col in range(COL_NUM + 2):
        num_matrix[0][col] = 0
    for row in range(ROW_NUM):
        num_matrix[row + 1] = {}
        num_matrix[row + 1][0] = 0
        for col in range(COL_NUM):
            grid_left = col * grid_width
            grid_top = row * grid_height
            grid_right = grid_left + grid_width
            grid_bottom = grid_top + grid_height
            grid_image = game_area[grid_top:grid_bottom, grid_left:grid_right]

            img = Image.fromarray(grid_image)
            img = transform(img).unsqueeze(0)
            img_ = img.to(device)
            outputs = mynet(img_)
            _, predicted = torch.max(outputs, 1)
            # print('this picture maybe :', predicted[0],classes[predicted[0]])

            num_matrix[row + 1][col + 1] = classes[predicted[0]]
            # cv2.imshow("./IMG/test/"+str(row)+","+str(col)+".jpg", grid_image)
            # cv2.waitKey(0)
            cv2.imwrite("./IMG/test/("+str(row+1)+","+str(col+1)+")("+ str(classes[predicted[0]])+").jpg", grid_image)
        num_matrix[row + 1][COL_NUM + 1] = 0

    num_matrix[ROW_NUM + 1] = {}
    for col in range(COL_NUM + 2):
        num_matrix[ROW_NUM + 1][col] = 0

    return num_matrix

# 相对坐标换算
def point_loc(x1, y1, x2, y2):
    global grid_height, grid_width
    # 200:788, 30:534]
    game_area_left = 30
    game_area_top = 200

    cx1 = game_area_left + (y1 + 0.5) * grid_width
    cy1 = game_area_top + (x1 + 0.5) * grid_height

    cx2 = game_area_left + (y2 + 0.5) * grid_width
    cy2 = game_area_top + (x2 + 0.5) * grid_height
    return cx1, cy1, cx2, cy2

# 数字矩阵检测块是否是背景块
def isBack(matrix, x, y):
    if matrix[x][y] == 0:
        return 1
    else:
        return 0

# 是否水平配对
def Dh(matrix, x1, y1, x2, y2):
    # 如果是同一个点, 返回错误
    if (x1 == x2) and (y1 == y2):
        return 0

    # 如果不在同一水平线上
    if (x1 != x2):
        return 0

    start_y = min(y1, y2)
    end_y = max(y1, y2)

    for j in range(start_y + 1, end_y):
        if isBack(matrix, x1, j) == 0:
            return 0

    # 不是以上情况，返回1（是）
    return 1

# 竖直检测
def Dv(matrix, x1, y1, x2, y2):
    # 如果是同一个点
    if (x1 == x2) and (y1 == y2):
        return 0

    # 如果不在同一竖直线上
    if (y1 != y2):
        return 0

    start_x = min(x1, x2)
    end_x = max(x1, x2)

    for i in range(start_x + 1, end_x):
        if isBack(matrix, i, y1) == 0:
            return 0

    return 1

# 是否通过一个拐角可以达到
def Dturn1(matrix, x1, y1, x2, y2):
    # 如果是同一个点
    if (x1 == x2) and (y1 == y2):
        return 0

    cx, cy, dx, dy = x1, y2, x2, y1

    if isBack(matrix, cx, cy) and ((Dh(matrix, x1, y1, cx, cy) and Dv(matrix, x2, y2, cx, cy)) or (Dh(matrix, cx, cy, x2, y2) and Dv(matrix, cx, cy, x1, y1))):
        return 1
    if isBack(matrix, dx, dy) and ((Dh(matrix, x1, y1, dx, dy) and Dv(matrix, x2, y2, dx, dy)) or (Dh(matrix, dx, dy, x2, y2) and Dv(matrix,dx, dy, x1, y1))):
        return 1

    return 0

# 是否通过两个拐角可以达到
def Dturn2(matrix, x1, y1, x2, y2):
    # 如果是同一个点
    if (x1 == x2) and (y1 == y2):
        return 0

    for i in range(0, ROW_NUM):
        for j in range(0, COL_NUM):
            # 跳过与A，B点不在同一水平竖直线的点
            if (i != x1) and (i != x2) and (j != y1) and (j != y2):
                continue
            # 跳过A点和B点
            if ((i == x1) and (j == y1)) or ((i == x2) and (j == y2)):
                continue
            # 跳过非背景点
            if isBack(matrix, i, j) == 0:
                continue
            # A经过一个转弯可以达到C，C水平竖直可以到达B
            if Dturn1(matrix, x1, y1, i, j) and (Dh(matrix, i, j, x2, y2) or Dv(matrix, i, j, x2, y2)):
                return 1
            # B经过一个拐弯可以到达C，C水平竖直可以到达A
            if Dturn1(matrix, i, j, x2, y2) and (Dh(matrix, x1, y1, i, j) or Dv(matrix, x1, y1, i, j)):
                return 1

    return 0

# 判断是否可消除
def removable(matrix, x1, y1, x2, y2):
    # 建立在两点相同的情况下
    if matrix[x1][y1] != matrix[x2][y2] or matrix[x1][y1] == 0 or matrix[x2][y2] == 0:
        return 0

    if Dh(matrix, x1, y1, x2, y2):
        print("-----------------------------------------")
        print("loc:(%d,%d) and (%d,%d)" % (x1, y1, x2, y2))
        print("opt type:'horizon'")
        return 1

    if Dv(matrix, x1, y1, x2, y2):
        print("-----------------------------------------")
        print("loc:(%d,%d) and (%d,%d)" % (x1, y1, x2, y2))
        print("opt type:'vertical'")
        return 1

    if Dturn1(matrix, x1, y1, x2, y2):
        print("-----------------------------------------")
        print("loc:(%d,%d) and (%d,%d)" % (x1, y1, x2, y2))
        print("opt type:'one turn'")
        return 1

    if Dturn2(matrix, x1, y1, x2, y2):
        print("-----------------------------------------")
        print("loc:(%d,%d) and (%d,%d)" % (x1, y1, x2, y2))
        print("opt type:'two turns'")
        return 1

    return 0

# 计算对应解
def solve_matrix(matrix, ):
    # 若不是零矩阵则一直循环
    #while numpy.where(matrix != 0)[0].shape[0] != 0:
    for index in range(1):
        for x1 in range(1, ROW_NUM + 1):
            for y1 in range(1, COL_NUM + 1):
                for x2 in range(1, ROW_NUM + 1):
                    for y2 in range(1, COL_NUM + 1):
                        if removable(matrix, x1, y1, x2, y2) == 1:
                            cx1, cy2, cx2, cy2 = point_loc(x1 - 1, y1-1, x2-1, y2-1)
                            outs.eliminate(cx1 , cy2, cx2, cy2)
                            matrix[x1][y1] = 0
                            matrix[x2][y2] = 0
                            print(matrix)

# 测试运行
if __name__ == "__main__":
    test_img = cv2.imread("./IMG/screen.jpg")
    game_area = test_img[200:788, 30:534]
    # cv2.imshow("game",game_area)
    # cv2.waitKey(0)
    matrix = createMatrix(game_area)
    #
    # # matrix = [
    # #     [0, 0, 0, 0, 0, 0, 0],
    # #     [0, 1, 2, 0, 3, 4, 0],
    # #     [0, 5, 6, 0, 7, 8, 0],
    # #     [0, 9, 10, 0, 11, 12, 0],
    # #     [0, 1, 12, 0, 10, 11, 0],
    # #     [0, 13, 6, 0, 13, 3, 0],
    # #     [0, 8, 2, 0, 9, 14, 0],
    # #     [0, 4, 7, 0, 5, 14, 0],
    # #     [0, 0, 0, 0, 0, 0, 0]
    # # ]
    for row in range(ROW_NUM+2):
        print(matrix[row])
    solve_matrix(matrix)
    print("done!")