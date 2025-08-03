import sys

# 打开文件，使用 'a' 模式表示追加到文件末尾
sys.stdout = open('output3.txt', 'a')

# 现在所有的 print 语句会写入到 'output.txt' 文件中


# 关闭文件
sys.stdout.close()
