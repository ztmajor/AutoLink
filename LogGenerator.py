import sys
import os
import datetime

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.path = os.path.join(self.path, "Log", filename)
        self.terminal = sys.stdout
        self.log = open(self.path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    # 在文件关闭前，将数据刷新到硬盘中
    def flush(self):
        pass


# sys.stdout = Logger(str(datetime.date.today()) + ".txt")
#
# print("123")