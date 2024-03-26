from model import Model
from view import View
from controllers import ControllersCalc

    # Создаем экземпляры классов
    model = Model()
    control = ControllersCalc(model)
    view = View()

    # Пример использования калькулятора комплексных чисел
    num1 = view.number_input("Введите первое комплексное число")
    num2 = view.number_input("Введите второе комплексное число")

    # Выполнение и вывод операций
    view.display('Сложение', control.sum_numbers(num1, num2))
    view.display('Вычитание', control.min_numbers(num1, num2))
    view.display('Умножение', control.mult_numbers(num1, num2))
    view.display('Деление', control.div_numbers(num1, num2))


# Точка входа в программу
while True:
    run_calculator()  # Запуск калькулятора
    n = input("Хотите ввести другие цифры? Y/N: ")
    if n.lower() != 'y':
        print("До свидания!")
        break
