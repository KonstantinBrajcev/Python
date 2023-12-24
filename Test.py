# list1 = [[1, 7, 8], [9, 7, 102], [6, 106, 105], [100, 99, 98, 103], [1, 2, 3]]
# maximum = 0
# for i in range(len(list1)):
#     if max(list1[i]) > maximum:
#         maximum = max(list1[i])

# print(maximum)


# list1 = [[1, 7, 8], [9, 7, 102], [102, 106, 105],
#          [100, 99, 98, 103], [1, 2, 3]]
# for i in range(len(list1)):
#     list1[i].reverse()
# print(list1)

# list1 = [[1, 7, 8], [9, 7, 102], [102, 106, 105],
#          [100, 99, 98, 103], [1, 2, 3]]
# total = 0
# counter = 0
# for i in range(len(list1)):
#     for j in range(len(list1[i])):
#         total += list1[i][j]
#         counter += 1
# print(total/counter)


# n = int(input())
# list = [1 * (i+1) for i in range(n)]
# for i in range(n):
#     print(list)


# n = int(input())
# list1 = [[1] for i in range(n)]
# for i in range(1, n):
#     list1[i] = [1 * (i+1) for i in range(i+1)]
# for i in range(n):
#     print(list1[i])


# def pack_sequences(string):
#     packed_list = []
#     current_char = string[0]
#     for char in string:
#         if char == current_char:
#             packed_list[-1].append(char)
#         else:
#             packed_list.append([char])
#             current_char = char
#     return packed_list


# input_string = input()
# packed_sequences = pack_sequences(input_string)
# print(packed_sequences)


# m = int(input())
# matrix = []
# sum = 0
# for i in range(m):
#     matrix.append(list(map(int, input().split())))
# for i in range(m):
#     for j in range(m):
#         if i == j:
#             sum += matrix[i][j]
# print(sum)


# n = int(input())
# matrix = []
# for i in range(n):
#     row = list(map(int, input().split()))
#     matrix.append(row)

# max_element = matrix[0][0]
# for i in range(n):
#     for j in range(i+1):
#         if matrix[i][j] >= max_element:
#             max_element = matrix[i][j]

# print(max_element)


# n = int(input())
# matrix = []
# for i in range(n):
#     row = list(map(int, input().split()))
#     matrix.append(row)
# upper_sum, right_sum, lower_sum, left_sum = matrix[0][0], matrix[0][0], matrix[0][0], matrix[0][0]
# for i in range(n):
#     for j in range(n):
#         if (i > j and i < n-1-j):
#             left_sum += matrix[i][j]
#         if (i < j and i > n-1-j):
#             right_sum += matrix[i][j]
#         if (i < j and i < n - 1 - j):
#             upper_sum += matrix[i][j]
#         if (i > j and i > n - 1 - j):
#             lower_sum += matrix[i][j]

# print('Верхняя четверть: ', upper_sum)
# print('Правая четверть: ', right_sum)
# print('Нижняя четверть: ', lower_sum)
# print('Левая четверть: ', left_sum)


# n = int(input())
# m = int(input())
# # Создание и заполнение матрицы
# matrix = []
# for i in range(n):
#     row = list(map(int, input().split()))
#     matrix.append(row)
# i, j = map(int, input().split())
# # Проверка на корректность введенных номеров столбцов
# if i < 0 or i >= m or j < 0 or j >= m:
#     print("Некорректные номера столбцов")
# else:
#     # Меняем местами столбцы
#     for k in range(n):
#         matrix[k][i], matrix[k][j] = matrix[k][j], matrix[k][i]
#     # Вывод измененной матрицы
#     for row in matrix:
#         print(" ".join(map(str, row)))


# # Считывание размеров матриц
# n, m = map(int, input().split())
# m1 = [list(map(int, input().split())) for _ in range(n)]
# print()
# m, k = map(int, input().split())
# m2 = [list(map(int, input().split())) for _ in range(m)]
# # Проверка возможности умножения матриц
# if len(m1[0]) != len(m2):
#     print("Умножение матриц невозможно")
# else:
#     # Создание результирующей матрицы
#     result = [[0] * k for _ in range(n)]
#     # Умножение матриц
#     for i in range(n):
#         for j in range(k):
#             for x in range(m):
#                 result[i][j] += m1[i][x] * m2[x][j]
#     # Вывод результирующей матрицы
#     for row in result:
#         print(' '.join(map(str, row)))
