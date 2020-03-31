import cv2
import numpy
import os
import time
from skimage.metrics import structural_similarity

ROW_NUM = 11
COL_NUM = 8

def img_sim(img1, img2):
    # 修改为统一大小为100 * 100
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))

    # 转化为灰度图
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    score = structural_similarity(img1, img2)
    return score

def get_Foreground(img, x1, y1, x2, y2):
    mask = numpy.zeros((img.shape[:2]), numpy.uint8)
    bgdModel = numpy.zeros((1, 65), numpy.float64)
    fgdModel = numpy.zeros((1, 65), numpy.float64)
    rect = (x1, y1, x2, y2)
    # 这里计算了5次
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # 关于where函数第一个参数是条件，满足条件的话赋值为0，否则是1。如果只有第一个参数的话返回满足条件元素的坐标。
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, numpy.newaxis]
    return img

def empty_hash(img):
    value = ""
    for i in range(10):
        c = img[i ,i]
        value += "%d,%d,%d," % (img[0], img[1], img[2])
    print(value)
    return value

def image_hash(img):
    value = ""
    for i in range(1, 5):
        c = img[i * 19, i * 19]
        value += "%d,%d,%d," % (c[0], c[1], c[2])
    return hash(value)

def createMatrix(game_area):
    img_matrix = {}

    grid_height = int(game_area.shape[0] / ROW_NUM)
    grid_width = int(game_area.shape[1] / COL_NUM)

    for row in range(ROW_NUM):
        img_matrix[row] = {}
        for col in range(COL_NUM):
            grid_left = col * grid_width
            grid_top = row * grid_height
            grid_right = grid_left + grid_width
            grid_bottom = grid_top + grid_height
            grid_image = game_area[grid_top:grid_bottom, grid_left:grid_right]
            #print(str(row) + "," + str(col) + ".jpg = ", image_id(grid_image))
            img_matrix[row][col] = grid_image
            # print(grid_left, grid_top, grid_right, grid_bottom)
            cv2.imwrite("./IMG/test/"+str(row)+","+str(col)+".jpg", grid_image)

    num_matrix = {}
    image_map = {}

    # for row in range(ROW_NUM):
    #     num_matrix[row] = {}
    #     for col in range(COL_NUM):
    #         this_image = img_matrix[row][col]
    #         this_image_hash = image_hash(this_image)
    #         if img_sim(this_image, img_matrix[0][4]) > 0.5:
    #             print(str(row) + "," + str(col) + "相似度 = ", img_sim(this_image, img_matrix[0][4]))
    #             num_matrix[row][col] = 0
    #             continue
    #         # image_map.setdefault(this_image_id, len(image_map) + 1)
    #         # num_matrix[row][col] = image_map.get(this_image_id)

    return num_matrix

# 相对坐标换算
# def point_loc(x1, y1, x2, y2):
#     cx1 = game_area_left + (from_col + 0.5) * grid_width
#     cy1 = game_area_top + (from_row + 0.5) * grid_height
#
#     cx2 = game_area_left + (to_col + 0.5) * grid_width
#     cy2 = game_area_top + (to_row + 0.5) * grid_height
#     return cx1, cy1, cx2, cy2

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
def solve_matrix(matrix):
    # 若不是零矩阵则一直循环
    #while numpy.where(matrix != 0)[0].shape[0] != 0:
    for index in range(1):
        for x1 in range(1, ROW_NUM - 1):
            for y1 in range(1, COL_NUM - 1):
                for x2 in range(1, ROW_NUM - 1):
                    for y2 in range(1, COL_NUM - 1):
                        if removable(matrix, x1, y1, x2, y2) == 1:
                            matrix[x1][y1] = 0
                            matrix[x2][y2] = 0
                            #print(matrix)

# 测试运行
if __name__ == "__main__":
    test_img = cv2.imread("./IMG/blocks/ground truth8.png")
    # img = cv2.imread("./test/0,0.jpg")
    # img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("img_before", img)
    game_area = test_img[183:1294, 45:861]
    # cv2.imshow("test_img", test_img)
    # img = get_Foreground(game_area, 183, 45, 1294, 861)
    # cv2.imshow("img", img)
    matrix = createMatrix(game_area)
    # matrix = [
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 2, 0, 3, 4, 0],
    #     [0, 5, 6, 0, 7, 8, 0],
    #     [0, 9, 10, 0, 11, 12, 0],
    #     [0, 1, 12, 0, 10, 11, 0],
    #     [0, 13, 6, 0, 13, 3, 0],
    #     [0, 8, 2, 0, 9, 14, 0],
    #     [0, 4, 7, 0, 5, 14, 0],
    #     [0, 0, 0, 0, 0, 0, 0]
    # ]
    # solve_matrix(matrix)
    print("done!")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(matrix)
