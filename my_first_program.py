# print('Скоро я', 'буду программировать', 'на языке', 'Python!')
# print('1', '2', '4', '8', '16')

# print('Как тебя зовут?')
# name = input()
# print('Привет,', name)


# print('Какой язык программирования ты изучаешь?')
# language = input()
# print(language, '- отличный выбор!')

# x = input ()
# print ("Привет,", x)

# x = int(input())
# print ('Следующее за числом', x, ' число:', (x+1))
# print ('Для числа', x , 'предыдущее число:', (x-1))

# x = int(input())
# y = int(input())
# z = int(input())
# k = int(input())
# print ((x+y+z+k)*3)


# x = int(input())
# y = int(input())
# print (x, '+',y, '=', x+y)
# print (x, '-',y, '=', x-y)
# print (x, '*',y, '=', x*y)


# a1 = int(input())
# d = int(input())
# n = int(input())
# print (a1+d*(n-1))


# x = int(input())
# print (x, x*2, x*3, x*4, x*5, sep='---')


# a = 82 // 3 ** 2 % 7
# print(a)


# b1 = int(input())
# q = int(input())
# n = int(input())
# print(b1*q**(n-1))


# n = int(input())
# print(n//100)


# y = int(input())
# x = int(input())
# print(x//y)
# print(x%y)


# x = int(input())
# print((x+1)//2)


# x = int(input())
# x = (x-1)//4
# print(x+1)


# x = int(input())
# print(x,'мин - это',x//60,'час',x%60,'минут')


# x=int(input())
# # print('Цифра в позиции тысяч равна', x//1000)
# # print('Цифра в позиции сотен равна', x//100%10)
# # print('Цифра в позиции десятков равна', x%100//10)
# # print('Цифра в позиции единиц равна', x%100%10)
# summ = (x//1000)+(x%100%10)
# razn = (x//100%10)-(x%100//10)
# if summ == razn:
#     print('ДА')
# else:
#     print('НЕТ')

# x=int(input())
# if x > 16:
#     print('Доступ разрешен')
# else:
#     print('Доступ запрещен')

# x=int(input())
# y=int(input())
# if x<y:
#     print(x)
# else:
#     print(y)

# a=int(input())
# if a <= 13:
#     print('детство')
# if 14 <= a <= 24:
#     print('молодость')
# if 25 <= a <= 59:
#     print('зрелость')
# if a >= 60:
#     print('старость')


# x=int(input())
# y=int(input())
# z=int(input())
# if x<0:
#     x=0
# if y<0:
#     y=0
# if z<0:
#     z=0
# print(x+y+z)


# x=int(input())
# if -30 <= x <= -2 or 7 <= x <= 25:
#     print('Принадлежит')
# else:
#     print('Не принадлежит')


# x=int(input())
# if (1000 <= x <= 9999) and (x % 7 == 0) or (x % 17 == 0):
#     print('YES')
# else:
#     print('NO')


# a=int(input())
# b=int(input())
# c=int(input())
# if a+b>c and a+c>b and b+c>a:
#     print('YES')
# else:
#     print('NO')


# a=int(input())
# if (a%4 == 0 and a%100 != 0) or a%400 == 0:
#     print('YES')
# else:
#     print('NO')


# a=int(input())
# b=int(input())
# c=int(input())
# d=int(input())
# if abs(c-a) <=1 and abs(d-b) <=1:
#     print('YES')
# else:
#     print('NO')


# a=int(input())
# b=int(input())
# if a>b:
#     print('YES')
# elif b>a:
#     print('NO')
# else:
#     print("Don't know")


# numbers = [2, 5, 13, 7, 6, 4]
# size = 6
# sum = 0
# avg = 0
# index = 0
# while index < size:
#     sum = sum + numbers[index]
#     index = index + 1
# avg = sum / size
# print(avg)


# a=int(input())
# b=int(input())
# c=int(input())
# if a==b==c:
#     print('Равносторонний')
# elif a!=b!=c and c!=a:
#     print('Разносторонний')
# else:
#     print('Равнобедренный')


# a=int(input())
# b=int(input())
# c=int(input())
# if c<a<b or b<a<c:
#     print(a)
# elif a<b<c or c<b<a:
#     print(b)
# else:
#     print(c)


# a=int(input())
# mes = [31,28,31,30,31,30,31,31,30,31,30,31]
# print(mes[a-1])


# a=int(input())
# if 64<=a<=69:
#     print('Полусредний вес')
# elif 64>a>=60:
#     print('Первый полусредний вес')
# else:
#     print('Легкий вес')


# a=int(input())
# b=int(input())
# c=input()
# znak = ['+','-','*','/']
# if c == '/' and b == 0:
#     print('На ноль делить нельзя!')
# elif c in znak:
#     if (c == znak[0]):
#         print(a+b)
#     if (c == znak[1]):
#         print(a-b)
#     if (c == znak[2]):
#         print(a*b)
#     if (c == znak[3]):
#         print(a/b)
# else:
#     print('Неверная операция')


# a = input()
# b = input()
# arr = ['красный', 'синий', 'желтый']
# if (a in arr) and (b in arr):
#     if a == b:
#         print(a)
#     if (a == arr[0] and b == arr[1]) or (a == arr[1] and b == arr[0]):
#         print('фиолетовый')
#     if (a == arr[0] and b == arr[2]) or (a == arr[2] and b == arr[0]):
#         print('оранжевый')
#     if (a == arr[1] and b == arr[2]) or (a == arr[2] and b == arr[1]):
#         print('зеленый')
# else:
#     print('ошибка цвета')


# n=int(input())
# if (0<=n<=36):
#     if n==0:
#         print('зеленый')
#     elif ((n%2==0) and ((1<=n<=10) or (19<=n<=28))) or ((n%2>0) and ((11<=n<=18) or (29<=n<=36))):
#         print('черный')
#     else:
#         print('красный')
# else:
#     print('ошибка ввода')


# a1=int(input())
# b1=int(input())
# a2=int(input())
# b2=int(input())

# # ТОЧКИ
# if (b1==a2 or b2==a1):
#     if b1==a2:
#         print(b1)
#     else:
#         print(b2)

# # СОВПАДЕНИЕ НАЧАЛА
# if (a1==a2) and (b1>b2):
#     print('286',a1, b2)
# if (a1==a2) and (b2>b1):
#     print('288',a1, b1)

# # ОДИН В ДРУГОМ
# if (a1>a2) and (b2>b1) :
#     print('292',a1, b1)
# if (a2>a1) and (b1>b2):
#     print('294',a2, b2)

# # СОВПАДЕНИЕ КОНЦА
# if (b1==b2) and (a1>a2):
#     print('298',a1, b1)
# if (b1==b2) and (a2>a1):
#     print('300',a2, b1)

# # СМЕЩЕНИЕ
# if (a2>a1) and (b2>b1) and (b1>a2):
#     print('304', a2, b1)
# if (a1>a2) and (b1>b2) and (a1<b2):
#     print('306',a1, b2)

# # ПОЛНОЕ СОВПАДЕНИЕ
# if (a1==a2) and (b1==b2):
#     print('310', a1, b1)

# # НЕСТЫКОВКА
# if (b1<a2) or (b2<a1):
#     print('пустое множество')


# x=int(input())
# if x%100//10 == 0 == x%100%10:
#     print('YES')
# else:
#     print('NO')

# x1, y1, x2, y2 = int(input()), int(input()), int(input()), int(input())
# if ((x1+y1+x2+y2)%2 == 0):
#     print('YES')
# else:
#     print('NO')


# old, mf = int(input()), input()
# if (10<=old<=15) and (mf == 'f'):
#     print('YES')
# else:
#     print('NO')


# a = int(input())
# if 1<=a<=10:
#     if a==1:
#         print('I')
#     if a==2:
#         print('II')
#     if a==3:
#         print('III')
#     if a==4:
#         print('IV')
#     if a==5:
#         print('V')
#     if a==6:
#         print('VI')
#     if a==7:
#         print('VII')
#     if a==8:
#         print('VIII')
#     if a==9:
#         print('IX')
#     if a==10:
#         print('X')
# else:
#     print('ошибка')


# a = int(input())
# if (a%2==0) and ((2<=a<=5) or (a>20)):
#     print('NO')
# if ((a%2==0) and (6<=a<=20) or (a%2!=0)):
#     print('YES')


# x1, y1, x2, y2 = int(input()), int(input()), int(input()), int(input())
# if (x1 - y1) == (x2 - y2) or (x1 + y1) == (x2 + y2):
#     print('YES')
# else:
#     print('NO')


# x1, y1, x2, y2 = int(input()), int(input()), int(input()), int(input())
# if (x1==x2 and y1!=y2) or (x1!=x2 and y1==y2) or abs(x1-x2) == abs(y1-y2):
#     print('YES')
# else:
#     print('NO')

# a, b = float(input()), float(input())
# print(1/2*a*b)

# s, v1, v2 = float(input()), float(input()), float(input())
# print(s/(v1+v2))

# a = float(input())
# if a!=0:
#     print(1/a)
# else:
#     print('Обратного числа не существует')


# a = float(input())
# print(5/9*(a-32))


# a = float(input())
# if a<=2:
#     print(a*10.5)
# else:
#     print(21+((a-2)*4))

# a = float(input())
# print(int(a*10)%10)

# a=float(input())
# b=int(a)
# print(a-b)


# a,b,c,d,e=int(input()), int(input()), int(input()), int(input()), int(input())
# max = max(a,b,c,d,e)
# min = min(a,b,c,d,e)
# print('Наименьшее число =', min)
# print('Наибольшее число =', max)

# a,b,c=int(input()), int(input()), int(input())
# max = (a,b,c)
# min = (a,b,c)
# print(min)
# print(a+b+c-min-max)
# print(max)


# x=int(input())
# a=x//100%10
# b=x%100//10
# c=x%10
# max = max(a,b,c)
# min = min(a,b,c)
# sred = (a+b+c-max-min)
# if max-min == sred:
#     print('Число интересное')
# else:
#     print('Число неинтересное')


# a=float(input())
# b=float(input())
# c=float(input())
# d=float(input())
# e=float(input())
# sum=abs(a)+abs(b)+abs(c)+abs(d)+abs(e)
# print(sum)

# p1=int(input())
# p2=int(input())
# q1=int(input())
# q2=int(input())
# print(abs(p1-q1)+abs(p2-q2))

# mystr = 'да'
# mystr = mystr + 'нет'
# mystr = mystr + 'да'
# print(mystr)


# str1 = '1'
# str2 = str1 + '2' + str1
# str3 = str2 + '3' + str2
# str4 = str3 + '4' + str3
# print(str4)


# x = str(input())
# if ('суббота' or 'воскресенье') in x:
#     print('YES')
# else:
#     print('NO')


# x = str(input())
# if '@' in x and '.' in x:
#      print('YES')
# else:
#     print('NO')


# x1, y1, x2, y2 = float(input()), float(input()), float(input()), float(input())
# from math import
# print( sqrt( (pow((x1-x2), 2))+(pow((y1-y2), 2) ) ) )


# r = float(input())
# from math import *
# print(pi * r * r)
# print(2 * pi * r)

# x, y = float(input()), float(input())
# from math import *
# print((x+y)/2)
# print(sqrt(x*y))
# print((2*x*y)/(x+y))
# print(sqrt(((x*x)+(y*y))/2))


# from math import *
# x = radians(float(input()))
# print(sin(x)+cos(x)+(tan(x)**2))

# from math import *
# x = float(input())
# print(floor(x)+ceil(x))


# from math import *
# a = float(input())
# b = float(input())
# c = float(input())
# D = (b*b) - (4*a*c)
# if D>0 or D==0:
#     x1=((-b)+(D**0.5))/(2*a)
#     x2=((-b)-(D**0.5))/(2*a)
#     if x1<x2:
#         print(x1)
#         print(x2)
#     else:
#         print(x2)
#         print(x1)
# if D==0:
#     print(((-b)+(D**0.5))/(2*a))
# if D<0:
#     print('Нет корней')


# import math
# n = int(input())
# a = float(input())
# s = (n*a**2)/(4*math.tan(math.pi/n))
# print(s)


# for i in range(10):
#     print("Python is awesome!")

# x = input()
# y = int(input())
# for i in range(y):
#     print(x)


# for i in range(6):
#     print('AAA')
# for i in range(4):
#     print('BBBB')
# print('E')
# for i in range(9):
#     print('TTTTT')
# print('G')


# n = int(input())
# for i in range(n):
#     print('Квадрат числа', i, 'равен', i*i)


# n = int(input())
# for i in range(n):
#     print('*'*(n-i))


# m,p,n=float(input()),float(input()),int(input())
# print(1, m)
# for i in range(n-1):
#     m = (m+(m*p/100))
#     print(i+2, m)

# m,n=int(input()),int(input())
# if m>n:
#     for i in range(m, n-1, -1):
#         print(i)
# if n>m:
#     for i in range(m, n+1, 1):
#         print(i)
# if m==n:
#     print(m)


# m,n=int(input()),int(input())
# for i in range(m, n+1,):
#     if (i%17 == 0) or (i%5 == 0 and i%3 == 0) or (i%10 == 9):
#         print(i)

# n=int(input())
# for i in range(1, 11):
#     print(n, 'x', i, '=', n*i)

# m,n=int(input()),int(input())
# count = 0
# for i in range (m, n+1):
#     if ((i**3)%10 == 4) or (((i**3)%10 == 9)):
#         count += 1
# print(count)

# m=int(input())
# count = 0
# for i in range (m):
#     n=int(input())
#     count += n
# print(count)

# import math
# n=int(input())
# sum = 0
# for i in range (n):
#     sum += (1/(i+1))
#     # print(sum)
# sum -= (math.log(n))
# print(sum)


# n=int(input())
# sum = 0
# for i in range (1, n+1):
#     if ((i*i)%10 == 5):
#         sum += i
# print(sum)


# n=int(input())
# sum = 1
# for i in range(n):
#     sum *= (i+1)
# print(sum)

# sum=1
# for i in range(1, 11):
#     x=int(input())
#     if (x!=0):
#         sum = sum*x
# print(sum)

# x=int(input())
# sum=0
# for i in range(1, x+1):
#     if (x%i==0):
#         sum += i
# print(sum)


# x=int(input())
# sum = 0
# for i in range(x):
#     if i%2==0:
#         sum -= i
#     else:
#         sum += i
# sum += ((-1)**(x+1))*x
# print(sum)


# x = int(input())
# max = 0
# premax = 0
# for i in range(x):
#     n = int(input())
#     if n > premax:
#         max = premax
#         max = n
#     elif n > max:
#         max = n
# print(max)
# print(premax)

# for i in range(1, 4):
#     for j in range(3, 5):
#         print(i + j, end='')


# n = int(input("Введите высоту треугольника: "))
# number = 1
# for i in range(1, n+1):
#     for j in range(1, i+1):
#         print(number, end=" ")
#         number += 1
#     print()

# n = 5
# # Инициализируем первые два числа последовательности
# fib1 = 1
# fib2 = 1
# if (n == 1):
#     print(fib1)
# else:
#     # Выводим первые два числа
#     print(fib1, fib2, end=' ')
# # Генерируем оставшиеся числа последовательности и выводим их
#     for i in range(1, n):
#         fib = fib1 + fib2
#         print(fib, end=' ')
#         fib1 = fib2
#         fib2 = fib

# flag = "YES"
# for i in range(1, 11):
#     x = int(input())
#     if (x % 2 != 0):
#         flag = "NO"
#         break
# print(flag)


# flag = True
# while (flag == True):
#     x = input()
#     if (x == "КОНЕЦ"):
#         flag = False
#     print(x)

# flag = True
# while (flag == True):
#     x = int(input())
#     if (x % 7 != 0):
#         flag = False
#     else:
#         print(x)


# flag = True
# sum = 0
# while (flag == True):
#     x = int(input())
#     if (x < 0):
#         flag = False
#     else:
#         sum += x
# print(sum)


# flag = True
# five = 0
# while (flag == True):
#     x = int(input())
#     if (x == 5):
#         five += 1
#     if (x > 5):
#         flag = False
# print(five)


# def min_coins(n):
#     coins = [25, 10, 5, 1]  # доступные номиналы монет
#     count = 0
#     for coin in coins:
#         while n >= coin:
#             n -= coin
#             count += 1
#     return count


# print(min_coins(int(input())))


# x = input()
# from tkinter import W


# number = '6898745'
# min = 0
# max = 0
# matrix = [int(digit) for digit in str(number)]
# for i in range(len(matrix)-1, -1, -1):
#     if matrix[i] < min:
#         min = matrix[i]
#     if matrix[i] > max:
#         max = matrix[i]
# print("Максимальная цифра равна ", max)
# print("Минимальная цифра равна ", min)

# сумму его цифр;
# количество цифр в нем;
# произведение его цифр;
# среднее арифметическое его цифр;
# его первую цифру;
# сумму его первой и последней цифры.
# number = '6898745'


# mx = -10000000  # add 00000
# s = 0
# for i in range(0, 10):
#     x = int(input())
#     if x < 0:
#         s -= x  # add -
#     if x > mx and x < 0:  # add
#         mx = x
# print(s)
# print(mx)
# if s == 0:
#     print("NO")  # add


# s = 0
# for i in range(0, 7):
#     n = int(input())
#     if n % 2 == 0:
#         s += n
# if s == 0:
#     print(0)
# else:
#     print(s)

# Решите уравнение в натуральных числах
# 28n + 30k + 31m = 365.
# m = (365 - (30 * k) - (28 * n)) / 31
# n = (365 - (30 * k) - (31 * m)) / 28
# k = (365 - (31 * m) - (28 * n)) / 30
# (28 * n) + (30 * k) + (31 * m) - 365 = 0

# for n in range(1, 365):
#     for m in range(1, 365):
#         for k in range(1, 365):
#             if ((28 * n) + (30 * k) + (31 * m) - 365 == 0):
#                 print('n = ', n)
#                 print('m = ', m)
#                 print('k = ', k)

# Имеется 100 рублей. Сколько быков, коров и телят можно купить на все эти деньги,
# если плата за быка – 10 рублей, за корову – 5 рублей, за теленка – 0.5 рубля и надо купить 100 голов скота?
# for n in range(1, 100):
#     for m in range(1, 100):
#         for k in range(1, 100):
#             if ((0.5 * n) + (5 * k) + (10 * m) - 100 == 0):
#                 if (m + n + k == 100):
#                     print('n = ', n)
#                     print('m = ', m)
#                     print('k = ', k)

# Леонард Эйлер сформулировал обобщенную версию Великой теоремы Ферма, предполагая, что по крайней мере
# n энных степеней необходимо для получения суммы, которая сама является энной степенью для n>2.
# Напишите программу для опровержения гипотезы Эйлера и найдите четыре положительных целых числа,
# сумма 5-х степеней которых равна 5-й степени другого положительного целого числа.
# Таким образом, найдите пять натуральных чисел a,b,c,d,e, удовлетворяющих условию:
# a5 + b5 + c5 + d5 = e5
# В ответе укажите сумму a+b+c+d+e.

# for e in range(144, 150):
#     print('Пошли считать...')
#     for a in range(1, 150):
#         for b in range(a, 150):
#             for c in range(b, 150):
#                 for d in range(c, 150):
#                     if ((a ** 5) + (b ** 5) + (c ** 5) + (d ** 5) == (e ** 5)):
#                         # if ((((a ** 5) + (b ** 5) + (c ** 5) + (d ** 5)) ** (1/5)) == e):
#                         print('a = ', a)
#                         print('b = ', b)
#                         print('c = ', c)
#                         print('d = ', d)
#                         print('e = ', e)
#                         z = a+b+c+d+e
#                         print(z)
#                         break
#     print('e не равно = ', e)

# n = int(input())
# number = 1
# for i in range(1, n+1):
#     for j in range(1, i+1):
#         print(number, end=' ')
#         number += 1
#     print()

# n = int(input())
# for i in range(1, n+2):
#     num = 1
#     for j in range(1, i+(i-2)):
#         if (j >= i-1):
#             print(num, end=' ')
#             num -= 1
#         if (j < i-1):
#             print(num, end=' ')
#             num += 1
#     print()

# str = 'пипипип'
# flag = True
# for i in range(0, round(len(str)/2), 1):
#     if (str[i] != str[-(i+1)]):
#         flag = False
# if flag == False:
#     print("NO")
# else:
#     print("YES")


# str = 'There is no such thing as an accident. What we call by that name is the effect of some cause which we do not see'
# print(str[2])
# print(str[len(str)-2])
# print(str[0:5])
# print(str[0:len(str)-2])
# print(str[0::2])
# print(str[1::2])
# print(str[::-1])
# print(str[-1:-len(str):-2])


# print(len(str))
# print(str*3)
# print(str[0:1])
# print(str[0:3])
# print(str[len(str)-3:len(str)])
# print(str[::-1])
# print(str[1:len(str)-1])


# str = 'Chris Alan'
# arr = str.split(' ')
# print(arr)
# if arr[0][0] != arr[0][0].upper() or arr[1][0] != arr[1][0].upper():
#     print("NO")
# else:
#     print("YES")


# number_str = str(input())
# # Вычисляем сумму цифр
# digit_sum = sum(int(digit) for digit in number_str)
# # Вычисляем количество цифр
# digit_count = len(number_str)
# # Вычисляем произведение цифр
# digit_product = 1
# for digit in number_str:
#     digit_product *= int(digit)
# # Вычисляем среднее арифметическое цифр
# digit_average = digit_sum / digit_count
# # Вычисляем первую цифру
# first_digit = int(number_str[0])
# # Вычисляем сумму первой и последней цифры
# last_digit = int(number_str[-1])
# sum_first_last_digit = first_digit + last_digit
# print(digit_sum)
# print(digit_count)
# print(digit_product)
# print(digit_average)
# print(first_digit)
# print(sum_first_last_digit)

# def sum_of_divisors(n):
#     divisors_sum = 0
#     for i in range(1, n+1):
#         if n % i == 0:
#             divisors_sum += i
#     return divisors_sum


# n = 'АааГГЦЦцТТттт'
# n = n.lower()
# print(f'Аденин: {n.count('а')}')
# print(f'Гуанин: {n.count('г')}')
# print(f'Цитозин: {n.count('ц')}')
# print(f'Тимин: {n.count('т')}')


# n = 'Глупая критика не так заметна, как глупая похвала.'
# n = n.lower()
# glas = 'ауоыиэяюёе'
# sum_glas = 0
# soglas = 'бвгджзйклмнпрстфхцчшщ'
# sum_soglas = 0
# for i in range(0, len(glas)):
#     if n.count(glas[i]):
#         sum_glas += n.count(glas[i])
# for i in range(0, len(soglas)):
#     if n.count(soglas[i]):
#         sum_soglas += n.count(soglas[i])
# print(f'Количество гласных букв равно {sum_glas}')
# print(f'Количество согласных букв равно {sum_soglas}')


# Декодер в двоичную систему исчисления
# n = int(input())
# ostatok = ''
# while (n >= 1):
#     if n == 1:
#         ostatok += str(n)
#         break
#     ostatok += str(n % 2)
#     n = n // 2
# print(ostatok[::-1])

# количество вхождений в строку
# n = 3
# all = 0
# for i in range(0, n):
#     s = input()
#     sum = 0
#     if s.count('11'):
#         sum += s.count('11')
#     print(sum)
#     if sum >= 3:
#         all += 1
# print(all)


# n = 'www.stepik.org'
# if n.endswith('.com') or n.endswith('.ru'):
#     print("YES")
# else:
#     print("NO")


# s = input()  # Ввод строки текста
# first_h = s.find('h')  # Поиск индекса первого вхождения 'h'
# last_h = s.rfind('h')  # Поиск индекса последнего вхождения 'h'
# # Удаление символов между первым и последним 'h'
# result = s[:first_h] + s[last_h+1:]
# print(result)  # Вывод результата

# a = int(input())
# b = int(input())
# str = ''
# for i in range(a, b):
#     str += (chr(i) + ' ')
# print(str)


# st = 'Hello world!'
# st = [symbol for symbol in st]
# s = ''
# for i in range(0, len(st)):
#     s += (str(ord(st[i])) + ' ')
# print(s)


# ДЕКОДИРУЕМ СТРОКИ
# z = 14
# n = 'fsfftsfufksttskskt'
# n = [symbol for symbol in n]
# stroka = ''
# for i in range(0, len(n)):
#     num = ord(n[i])-z
#     if num < 97:
#         num = 122 - (96 - num)
#     stroka += str(chr(num))
# print(stroka)


# abc = 'abcdefghijklmnopqrstuvwxyz'
# z = [symbol for symbol in abc]
# for i in range(1, len(abc)):
#     z[i] = z[i]*(i+1)
# print(z)


# n = int(input())
# z = [int(input()) for i in range(n)]
# z = [z[i] + z[i+1] for i in range(n-1)]
# print(z)


# n = int(input())
# z = [int(input()) for i in range(n)]
# del z[1::2]
# print(z)


# n = 5
# z = [input() for i in range(n)]
# k = 3
# p = ''
# for i in range(len(z)):
#     if len(z[i]) >= k-1:
#         p += z[i][k-1]
# print(p)


# n = int(input())
# z = [input() for i in range(n)]
# z = list(''.join(z))
# print(z)


# numbers = [1, 78, 23, -65, 99, 9089, 34, -32, 0, -67, 1, 11, 111]
# z = [number**2 for number in numbers]
# print(z)


# n = int(input())
# z = [int(input()) for i in range(n)]
# print(*z, sep='\n')
# print()
# z1 = []
# for i in range(n):
#     z1.append((z[i]**2)+(2*z[i])+1)
# print(*z1, sep='\n')

# Удаление MAX и MIN
# n = int(input())
# z = [int(input()) for _ in range(n)]
# z.remove(max(z))
# z.remove(min(z))
# print(*z, sep='\n')


# ВЫВОДИМ УНИКАЛЬНЫЕ ЭЛЕМЕНТЫ
# s = [input() for _ in range(int(input()))]
# print(*[s[i] for i in range(len(s)) if s[:i].count(s[i]) == 0], sep="\n")


# ЗАПРОС В Н СТРОК
# s = [input() for _ in range(int(input()))]
# print('---')
# zap = input()
# print('---')
# print(*[s[i].lower() for i in range(len(s)) if zap in s[i].lower()], sep='\n')


# s = [int(input()) for _ in range(int(input()))]
# print(*[s[i] for i in range(len(s)) if s[i] < 0], sep='\n')
# print(*[s[i] for i in range(len(s)) if s[i] == 0], sep='\n')
# print(*[s[i] for i in range(len(s)) if s[i] > 0], sep='\n')


# print(*input().split(), sep='\n')


# n = 'Владимир Семенович Высоцкий'
# z = n.split()
# print(f'{z[0][:1]}.{z[1][:1]}.{z[2][:1]}.')


# Разделитель
# n = 'C:\Windows\System32\calc.exe'
# n = n.split('\\')
# print(*n, sep='\n')


# s = input()
# s = s.split(' ')
# for i in range(0, len(s)):
#     print('+'*int(s[i]))


# ПРОверка IP адреса
# n = input().split('.')
# for i in range(len(n)):
#     if int(n[i]) > 256 or int(n[i]) < 0:
#         print("NO")
#         break
# else:
#     print("YES")

# Меняем Максимальный и минимальный
# s = '10 9 8 7 6 5 4 3 2 1'.split(' ')
# num = [int(number) for number in s]
# min_id = num.index(min(num))
# max_id = num.index(max(num))
# num[min_id], num[max_id] = num[max_id], num[min_id]
# print(*num)


# text = 'William Shakespeare was born in the town of Stratford, England, in the year 1564. When he was a young man, Shakespeare moved to the city of London, where he began writing plays. His plays were soon very successful, and were enjoyed both by the common people of London and also by the rich and famous. In addition to his plays, Shakespeare wrote many short poems and a few longer poems. Like his plays, these poems are still famous today.'
# s = text.lower().split(' ')
# num = s.count('a') + s.count('an') + s.count('the')
# print(num)

# Удаление лишних коментов в коде
# z = [input() for _ in range(0, int(input()[1:]))]
# for i in range(len(z)):
#     if '#' in z[i]:
#         z[i] = z[i][:z[i].index('#')].rstrip()
# print(*z, sep='\n')


# Таблица ПАЛИНДРОВМОВ
# palindromes = [i for i in range(100, 1001) if str(i)[0] == str(i)[-1]]
# print(palindromes)

# Пузырьковая сотрировка
# a = [78, -32, 5, 39, 58, -5, -63, 57, 72, 9, 53, -1, 63, -97, -21, -94, -47, 57, -8, 60, -23, -72, -22, -79, 90, 96, -41, -71, -48, 84, 89, -96, 41, -16, 94, -60, -64, -39, 60, -14, -62, -19, -3, 32, 98, 14, 43, 3, -56,
#      71, -71, -67, 80, 27, 92, 92, -64, 0, -77, 2, -26, 41, 3, -31, 48, 39, 20, -30, 35, 32, -58, 2, 63, 64, 66, 62, 82, -62, 9, -52, 35, -61, 87, 78, 93, -42, 87, -72, -10, -36, 61, -16, 59, 59, 22, -24, -67, 76, -94, 59]

# n = len(a)
# for i in range(0, n):
#     k = i
#     for j in range(i+1, n):
#         if a[j] < a[k]:
#             k = j
#     a[k], a[i] = a[i], a[k]

# print(a)

# def merge(z):

#     return z


# n = int(input())
# sum = 0
# while (n > 1):
#     sum += 1
#     n = n / 2
# print(sum)

# def caesar(alphabet):
#     text = input('my name is Python!')
#     shift = ii("Shift: ")

#     def get_char(char, alphabet_, shift_):
#         if char.isalpha():
#             i = 0
#             if char.isupper():
#                 i = 1
#             return alphabet_[i][(alphabet_[i].index(char) + shift_) % len(alphabet_[0])]
#         return char

#     shifted = "".join([get_char(char, alphabet, shift) for char in text])
#     print(shifted)


# def english_alphabet():
#     return "".join([chr(char) for char in range(ord("a"), ord("z") + 1)])


# def ii(message=""):
#     return int(input(message))


# caesar([english_alphabet(), english_alphabet().upper()])


# ОПРЕДЕЛЕНИЕ ЧЕТВЕРТИ ВВОДА КООРДИНАТ
# n = int(input())
# x = []
# y = []
# num1 = num2 = num3 = num4 = 0
# for i in range(n):
#     z = (input().split())
#     x.append(int(z[0]))
#     y.append(int(z[1]))
# for i in range(n):
#     if x[i] > 0 and y[i] > 0:
#         num1 += 1
#     if x[i] < 0 and y[i] > 0:
#         num2 += 1
#     if x[i] < 0 and y[i] < 0:
#         num3 += 1
#     if x[i] > 0 and y[i] < 0:
#         num4 += 1
# print(f'Первая четверть: {num1}')
# print(f'Вторая четверть: {num2}')
# print(f'Третья четверть: {num3}')
# print(f'Четвертая четверть: {num4}')
