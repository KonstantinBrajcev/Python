class ComplexNumber:
    def __init__(self, real, imaginary):
        self.real = real        # Инициализация РЕАЛЬНОЙ части числа
        self.imaginary = imaginary  # Инициализация МНИМОЙ части числа

    def __str__(self):  # ВЫВОД результата
        if self.imaginary != 0 and self.real != 0:  # С мнимой частью
            if self.imaginary > 0:
                return f"{self.real}+{self.imaginary}i"
            else:
                return f"{self.real}{self.imaginary}i"
        elif self.imaginary == 0:  # БЕЗ мнимой части
            return f"{self.real}"
        elif self.real == 0:  # БЕЗ реальной части
            return f"{self.imaginary}i"

    def __add__(self, other):  # СЛОЖЕНИЕ комплексных чисел
        return ComplexNumber(self.real + other.real, self.imaginary + other.imaginary)

    def __sub__(self, other):  # ВЫЧИТАНИЕ комплексных чисел
        return ComplexNumber(self.real - other.real, self.imaginary - other.imaginary)

    def __mul__(self, other):  # УМНОЖЕНИЕ комплексных чисел
        return ComplexNumber(self.real * other.real - self.imaginary * other.imaginary, self.real * other.imaginary + self.imaginary * other.real)

    def __truediv__(self, other):  # ДЕЛЕНИЕ комплексных чисел
        denominator = other.real**2 + other.imaginary**2
        real_part = (self.real*other.real + self.imaginary *
                     other.imaginary) / denominator
        imaginary_part = (self.imaginary*other.real -
                          self.real*other.imaginary) / denominator
        return ComplexNumber(real_part, imaginary_part)
