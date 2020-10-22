# weight = 35.5
# stuID = 9000
# print("今年%.2f, %03d" % (weight, stuID))
# print(f'今年{weight}, {stuID + 1}')
# print("hello\nworld")
# print("hello world", end="...")
# print("hello myworld")

# weight = input("input weight: ")
#
# # print("weight: %s" % weight)
# print(f'weight: {weight}')
# print(type(weight))

# age = int(input('input age: '))
#
# if 0 <= age <= 19:
#     print(f'{age}')

# i = 1
# while i < 9:
#     j = 1
#     while j <= i:
#         print(f'{j} * {i} = {j * i}', end='\t')
#         j += 1
#     print()
#     i += 1

# i = 0
# while i < 5:
#     if i == 3:
#         i += 1
#         continue
#     print(i)
#     i += 1
#
# else:
#     print(6)

# print("i'm Tome")
# print('i\'m Tome')
# print(f'{"Tome"}')
#
# """
#     字符串是不可变数据类型
# """
# a = '12345678910'
# new_strs = a.split('6', 4)
# print(new_strs)
#
# List = ['aa', 'bb', 'cc']
# del List[0:2]
# print(List)

# import random
#
# teacher = ['1', '2', '3', '4', '5', '6', '7', '8']
#
# offices = [[], [], []]
#
# for item in teacher:
#     offices[random.randint(0, 2)].append(item)
#
# print(offices)

# s1 = {'a', 'b', 'c'}
# s1.update('efg')
# print(s1)


# s1 = 'abc'
# del s1[0]
# print(s1)

# list = ['a', 'b', 'c', 'd', 'e']
# for index, item in enumerate(list):
#     print(f'list[{index}] = {item}');

# list1 = ['name', 'age', 'sex']
# list2 = ['xiaoming', 12, 'man']
#
# dict = {list1[i]:list2[i] for i in range(len(list2))}
#
# print(dict)

# counts = {'a': 100, 'b': 150, 'c': 200, 'd': 250}
# # result = {key:value for key, value in counts.items() if value > 100}
# # print(result)
#
#
# for index, item in enumerate(counts.items()):
#     print(index, end='\t')
#     print(item[0], end='\n')
# set1 = set('12345')
# print(set1)

# def info_print(info: str):
#     """
#
#     :param info:
#     :return: nan
#     """
#     print(info)
#
#
# help(info_print)
# def sum_result(a: int, b: int, c: int):
#     return a + b + c
#
#
# def ave_result(a: int, b: int, c: int):
#     sum = sum_result(a, b, c)
#     return sum // 3
#
#
# print(ave_result(1, 2, 3))

# a = "hello world"
#
#
# def info_print(**kwargs):
#     # print(args[0])
#     # print(args[1])
#     for key, value in kwargs.items():
#         print(f'key = {key}\tvalue = {value}')
#
#
# info_print(str1="hello", str2="world")

# dict = {'a': '1', 'b': '2'}
# s1, s2 = dict
# print(s1, dict[s1], s2, dict[s2])

a, b = 1, 2
a, b = b, a
print(a, b)
