# def read_numbers_from_file(filename='random_numbers.txt'):

with open('random_numbers.txt', 'r') as file:
    lines = file.readlines()
    numbers = [float(line.strip()) for line in lines]
    # return numbers

for i in range(10):
    print(numbers[i])