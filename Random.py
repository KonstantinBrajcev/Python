import random
import os


def is_valid(z, n):
    if z > 0 and z < n+1:
        return z
    else:
        print(f'Введите целое число от 1 до {n}?')
        return is_valid(int(input()), n)


def game():
    os.system('clear')
    print('='*10 + ' Добро пожаловать в числовую угадайку! ' + '='*10)
    z = int(input('Угадываем число от 0 до N. Введите число N : '))
    n = random.randrange(0, z+1)
    num = is_valid(int(input('А теперь попробуй угадать число : ')), z)
    pop = 1
    while (num != n):
        if num > n:
            num = is_valid(int(input('N - Меньше. Попробуй еще раз : ')), z)
        else:
            num = is_valid(int(input('N - Больше. Попробуй еще раз : ')), z)
        pop += 1
    print('='*10+f' Поздравляю! Вы угадали число за {pop} попыток '+'='*10)
    new_or_no()


def new_or_no():
    new = input('Хотите СЫГРАТЬ ЕЩЕ РАЗ? (д/н) : ')
    while True:
        if (new == 'д'):
            game()
            break
        elif (new == 'н'):
            print('='*10+' Спасибо, что играли в числовую угадайку. '+'='*10)
            break
        else:
            print('Некорректный ввод. Введите "д" или "н".')


game()
