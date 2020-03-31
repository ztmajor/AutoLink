import cv2
import time
import win32gui, win32con

import ImgFetcher as imgf
import ImgProcessor as imgp
import OutputSimulator as outs

# 参数
THRESHOLD = 0.8

# 句柄
FrameClass = "LDPlayerMainFrame"
FrameTitle = "雷电模拟器"

# 图像
app_img = [cv2.imread("./IMG/app.JPG"), "APP"]
begin_img = [cv2.imread("./IMG/djllk.JPG"), "单机连连看"]
model_img = [cv2.imread("./IMG/FastMode.JPG"), "极速模式"]
round_img = [cv2.imread("./IMG/Turn1.JPG"), "Round 1"]
back_img = [cv2.imread("./IMG/background.jpg"), "背景图"]
next_img = [cv2.imread("./IMG/NextTurn.JPG"), "下一关"]


if __name__ == "__main__":
    #cv2.namedWindow('screen')

    PointList = [begin_img, model_img]

    # 获取后台窗口的句柄，注意后台窗口不能最小化(使用SPY++获取类名和窗口标题)
    pWin = win32gui.FindWindow(FrameClass, FrameTitle)
    if pWin == 0:
        exit(-1)
    #cWin = win32gui.FindWindowEx(pWin, 0, "subWin", None)
    win32gui.ShowWindow(pWin, win32con.SW_SHOWNORMAL)
    win32gui.SetForegroundWindow(pWin)

    # 获取窗口大小
    _left, _top, _right, _bottom = win32gui.GetWindowRect(pWin)
    width = _right - _left
    height = _bottom - _top

    for now_img in PointList:
        time.sleep(0.5)  # 等待1s
        screen = imgf.getScreen(pWin, width, height)
        click_point = imgf.match_img(screen, now_img[0], THRESHOLD)
        cv2.imwrite(now_img[1] + ".jpg", screen)
        print("click", now_img[1])
        print("click_point = ", click_point)
        outs.doClick(click_point[0], click_point[1])

        cv2.imshow("screen", screen)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    #screen = imgf.getScreen(pWin, width, height)
    #matrix = imgp.createMatrix(screen)