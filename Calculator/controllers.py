class ControllersCalc:  # Контроллер
    def __init__(self, model):
        self.model = model

    def sum_numbers(self, num1, num2):
        # Возвращаем сумму
        return self.model.sum(num1, num2)

    def min_numbers(self, num1, num2):
        # Возвращаем разницу
        return self.model.min(num1, num2)

    def mult_numbers(self, num1, num2):
        # Возвращаем произведение
        return self.model.mult(num1, num2)

    def div_numbers(self, num1, num2):
        # Возвращаем частное
        return self.model.div(num1, num2)
