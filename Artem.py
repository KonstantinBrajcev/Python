m = int(input("Введите колличество цифр : "))
n = int(input("Начальная цифра : "))


def Print(m, n):
    if (m == 0):
        return
    else:
        print((str(n) + " + ") * (m-1) + str(n))
        print("|  /" * (m-1))
        Print(m-1, n*2)


Print(m, n)
