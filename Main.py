import cv2
import datetime
import time
import sys
import win32gui, win32con
import numpy

import ImgFetcher as imgf
import ImgProcessor as imgp
import OutputSimulator as outs
import LogGenerator as logg
from MYNet import Net
from PIL import Image

import matplotlib.pyplot as plt

# 句柄
FrameClass = "LDPlayerMainFrame"
FrameTitle = "雷电模拟器"

# 图像
app_img = [cv2.imread("./IMG/app.JPG"), "APP"]
begin_img = [cv2.imread("./IMG/djllk.JPG"), "单机连连看"]
model_img = [cv2.imread("./IMG/EndlessMode.JPG"), "无尽模式"]
round_img = [cv2.imread("./IMG/Turn9.JPG"), "Round 9"]
back_img = [cv2.imread("./IMG/background.jpg"), "背景图"]
next_img = [cv2.imread("./IMG/NextTurn.JPG"), "下一关"]

ROW_NUM = 7
COL_NUM = 6

if __name__ == "__main__":
    # 日志
    sys.stdout = logg.Logger(str(datetime.date.today()) + ".txt")
    #cv2.namedWindow('screen')

    PointList = [begin_img, model_img]

    # 获取后台窗口的句柄，注意后台窗口不能最小化(使用SPY++获取类名和窗口标题)
    hwnd = win32gui.FindWindow(FrameClass, FrameTitle)
    if hwnd == 0:
        exit(-1)
    # cWin = win32gui.FindWindowEx(pWin, 0, "subWin", None)
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetForegroundWindow(hwnd)

    for now_img in PointList:
        time.sleep(0.5)  # 等待0.5s
        screen = imgf.getAppScreen(hwnd)[0]
        click_point = imgf.match_img(screen, now_img[0])
        # cv2.imwrite(now_img[1] + ".jpg", screen)
        print("click", now_img[1])
        print("click_point = ", click_point)
        outs.doClick(click_point[0], click_point[1])

    # 玩3关,一关3个阶段
    for i in range(0, 3*3):
        t = time.time()
        print("Round", int(i/3+1), ", Part", i%3+1, ":")
        time.sleep(4)
        # screen为PIL格式图片
        screen = imgf.getAppScreen(hwnd)[1]
        game_area = screen.crop((30, 200, 534, 788))
        # game_area = screen[200:788, 30:534]

        # plt.figure("game-area")
        # plt.imshow(game_area)
        # plt.show()
        matrix = imgp.createMatrix(game_area)

        for row in range(ROW_NUM + 2):
            print(matrix[row])

        # 求解
        while not numpy.all(matrix == 0):
            # 每循环一边重新生成矩阵（有的图每点击一次会改变）
            time.sleep(2)       # 避免截图到连击数
            screen = imgf.getAppScreen(hwnd)[1]
            game_area = screen.crop((30, 200, 534, 788))
            matrix = imgp.createMatrix(game_area)
            # 难以解决一些点击后改变地图的图
            imgp.quick_solve_matrix(matrix)
            # # 可解决点击改变问题，但速度过慢，难以过关
            # imgp.solve_matrix(matrix)

        # 无尽模式每3个阶段为一关
        if (i+1) % 3 == 0:
            time.sleep(2)  # 等待2s，跳过动画
            screen = imgf.getAppScreen(hwnd)[0]
            click_point = imgf.match_img(screen, next_img[0])
            cv2.imwrite(str(i) + next_img[1] + ".jpg", screen)
            print("click", next_img[1])
            print("click_point = ", click_point)
            outs.doClick(click_point[0], click_point[1])

        t1 = time.time()
        print("用时：", t1 - t)
        print("done!")

