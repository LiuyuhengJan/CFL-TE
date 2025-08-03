import random

numbers = list(range(100))  # 创建一个包含0-99的列表
random.shuffle(numbers)  # 使用random.shuffle打乱列表顺序

while numbers:  # 当列表不为空时循环
    number = numbers.pop()  # 从列表末尾移除一个元素并打印
    print(number)