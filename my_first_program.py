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


# a=input()
# b=input()
# arr = ['красный','синий','желтый']
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


# x=int(input())
# max=0
# premax=0
# for i in range(x):
#     n=int(input())
#     if n>premax:
#         max=premax
#         max=n
#     elif n>max:
#         max=n
# print(max)
# print(premax)

# Калькулятор
while True: print(eval(input('')))