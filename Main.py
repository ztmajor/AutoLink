import cv2
import time
import win32gui, win32con

import ImgFetcher as imgf
import ImgProcessor as imgp
import OutputSimulator as outs
from alexnet import Net

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
        time.sleep(0.5)  # 等待1s
        screen = imgf.getAppScreen(hwnd)
        click_point = imgf.match_img(screen, now_img[0])
        # cv2.imwrite(now_img[1] + ".jpg", screen)
        print("click", now_img[1])
        print("click_point = ", click_point)
        outs.doClick(click_point[0], click_point[1])

    time.sleep(4)
    screen = imgf.getAppScreen(hwnd)
    game_area = screen[200:788, 30:534]
    # cv2.imwrite("screen.jpg", screen)
    # cv2.imwrite("game_area.jpg", game_area)

    matrix = imgp.createMatrix(game_area)

    for row in range(ROW_NUM + 2):
        print(matrix[row])
    imgp.solve_matrix(matrix)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("done!")