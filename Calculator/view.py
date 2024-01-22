import logging
from complex_number import ComplexNumber


class View:
    def setup_logging():  # формат записи лог-файлов
        logging.basicConfig(filename='logging.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    setup_logging()       # Ведем логирование операций

    def number_input(self, prompt):  # Вводим комплексные числа
        while True:
            try:
                input_str = input(prompt + " (в формате a+bi): ")
                real, imag_str = input_str.split('+')
                imag = imag_str[:-1]   # Удаляем символ 'i' в конце строки
                return ComplexNumber(float(real), float(imag))
            except ValueError:
                print("НЕВЕРНЫЙ ФОРМАТ! Введите число в правильном формате.")

    def display(self, text, result):
        # Сообщение для вывода и логирования
        message = f"{text} {result}"
        print(message)
        logging.info(message)
