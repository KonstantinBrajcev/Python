# -------------------------------
# Задача №1
# n = 123
# n1 = int(n/100)
# print(n1)
# n2 = int(n/10)-int(n1*10)
# print(n2)
# n3 = int(n % 10)
# print(n3)
# res = n1+n2+n3
# print(res)

# Задача №2
# n = 6
# n2 = int(n*4)
# print(int(n/6), int(n2/6), int(n/6))

# Задача №3
# n = 357753
# n1 = int(((n/1000))/100)
# print(n1)
# n2 = int(((n/1000))/10)-int(n1*10)
# print(n2)
# n3 = int(((n/1000)) % 10)
# print(n3)
# res1 = n1+n2+n3
# print(res1)
# z1 = int((n % 1000)/100)
# print(z1)
# z2 = int(((n % 1000)/10) - int(z1*10))
# print(z2)
# z3 = int((n % 1000) % 10)
# print(z3)
# res2 = z1+z2+z3
# print(res2)

# if (res1 == res2):
#     print("yes")
# else:
#     print("no")


# Задача №4
# a=3
# b=2
# c=4
# if c == 1:
#     print("no")
# elif c % a == 0 or c % b == 0:
#     print("yes")
# elif c < a * b and c not in [a, b]:
#     print("yes")
# else:
#     print("no")

# ---------Орел и решка-------------
# Дана строка текста, состоящая из букв русского алфавита "О" и "Р". Буква "О" – соответствует выпадению Орла, а буква "Р" – соответствует выпадению Решки.
# Напишите программу, которая подсчитывает наибольшее количество подряд выпавших Решек.
# На вход программе подается строка текста, состоящая из букв русского алфавита "О" и "Р".
# Sample Input 1:
# ОРРОРОРООРРРО
# Sample Output 1:
# 3
# ------------------------------
# n = "ОРРРРОРООРРРО"  # ответ - 3
# sum = 0                     # сумма последовательных элементов строки
# max_count = 0
# for char in n:        # перебор элементов строки
#     if char == 'Р':      # если выпала "РЕШКА"
#         sum += 1            # складываем
#         max_count = max(max_count, sum)
#     else:
#         sum = 0            # обнуляем счетчик
# print(max_count)


# Задача №1
# На столе лежат n монеток. Некоторые из монеток лежат вверх решкой, а некоторые – гербом.
# задача - определить минимальное количество монеток, которые нужно перевернуть, чтобы все монетки лежали одной и той же стороной вверх.
# Входные данные: программе подается список coins, где coins[i] равно 0, если i-я монетка лежит гербом вверх, и равно 1, если i-я монетка лежит решкой вверх. Размер списка не превышает 1000 элементов.
# Выходные данные: Программа должна вывести одно целое число - минимальное количество монеток, которые нужно перевернуть.
# coins = [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0,
#          0, 1, 0, 0, 1, 0, 1, 0]  # ответ 12 - 8
# # -----1------
# # print(coins.count(0) if coins.count(0) < coins.count(1) else coins.count(1))
# # -----2-----
# if coins.count(0) > coins.count(1):
#     print(coins.count(1))
# else:
#     print(coins.count(0))


# Задача №2
# Петя и Катя – брат и сестра. Петя – студент, а Катя – школьница. Петя помогает Кате по математике.
# Петя задумывает два натуральных числа X и Y (X,Y≤1000), а Катя должна их отгадать. Для этого Петя делает две подсказки.
# Он называет сумму этих чисел S и их произведение P. Помогите Кате отгадать задуманные Петей числа.
# Примечание: числа S и P задавать не нужно, они будут передаваться в тестах.
# В результате вы должны вывести все возможные варианты чисел X и Y через пробел.
# x, y = 8, 5  # загадано
# s, p = 15, 56  # извесно
# for i in range(1, s):
#     if i * (s - i) == p:
#         print(i, s - i)

# Задача №3 - Требуется вывести все целые степени двойки (т.е. числа вида 2k), не превосходящие числаN.
# n = 10
# i = 0
# while (2 ** i) <= n:
#     print(2 ** i)
#     i += 1


# Задача 1 - Требуется вычислить, сколько раз встречается некоторое число k в массиве list_1.
# Найдите количество и выведите его.
# list_1 = [1, 2, 3, 2, 5]
# k = int(input("Какое число искать: "))
# count = 0
# for i in range(0, len(list_1)):
#     if list_1[i] == k:
#         count += 1
# print(count)


# Задача 2 - Требуется найти в массиве list_1 самый близкий по величине элемент к заданному числу k и вывести его.
# list_1 = [1, 2, 3, 4, 5, 6, 7, 9, 12, 22, 55, 78]
# k = int(input("Какое число искать: "))
# num = list_1[0]
# separate = abs(list_1[0]-k)
# for i in range(0, len(list_1)):
#     if abs(list_1[i] - k) < separate:
#         num = list_1[i]
#         separate = abs(list_1[i]-k)
# print(num)

# ---------Сколько различных цифр в массиве
# numbers = [1, 1, 2, 0, -1, 3, 4, 4]
# count = len(set(numbers))
# print(count)


# --------Сдвиг массива на N элементов
# sequence = [1, 2, 3, 4, 5]
# print(sequence)
# k = 3
# result = sequence[-k:] + sequence[:-k]
# print(result)

# В настольной игре Скрабл (Scrabble) каждая буква имеет определенную ценность.
# В случае с английским алфавитом очки распределяются так:
# А русские буквы оцениваются так:
# А, В, Е, И, Н, О, Р, С, Т – 1 очко;
# Д, К, Л, М, П, У – 2 очка;
# Б, Г, Ё, Ь, Я – 3 очка;
# Й, Ы – 4 очка;
# Ж, З, Х, Ц, Ч – 5 очков;
# Ш, Э, Ю – 8 очков;
# Ф, Щ, Ъ – 10 очков.
# Напишите программу, которая вычисляет стоимость введенного пользователем слова k и выводит его.
# Будем считать, что на вход подается только одно слово, которое содержит либо только английские, либо только русские буквы.
# # Пример:
# elements = [['A', 'E', 'I', 'O', 'U', 'L', 'N', 'S', 'T', 'R', 'А', 'В', 'Е', 'И', 'Н', 'О', 'Р', 'С', 'Т'],  # – 1 очко;
#             ['D', 'G', 'Д', 'К', 'Л', 'М', 'П', 'У'],  # – 2 очка;
#             ['B', 'C', 'M', 'P', 'Б', 'Г', 'Ё', 'Ь', 'Я'],  # – 3 очка;
#             ['F', 'H', 'V', 'W', 'Y', 'Й', 'Ы'],  # – 4 очка;
#             ['K', 'Ж', 'З', 'Х', 'Ц', 'Ч'],  # – 5 очков;
#             ['J', 'X', 'Ш', 'Э', 'Ю'],  # – 8 очков;
#             ['Q', 'Z', 'Ф', 'Щ', 'Ъ']]  # – 10 очков.
# k = input("Введите текст: ")
# arr = list(k.upper())
# summ = 0
# for i in range(0, len(arr)):
#     for j, row in enumerate(elements):
#         if arr[i] in row:
#             if j == 5:
#                 summ += (j+3)
#             elif j == 6:
#                 summ += (j+4)
#             else:
#                 summ += (j+1)
# print("Стоимость слова: ", summ)
