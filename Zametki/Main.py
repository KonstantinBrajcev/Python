import os
from Command import Command

os.system('clear')  # Очистка консоли
while True:
    file_path = 'notes.json'
    print("-----------МЕНЮ----------",
          "1 -> ДОБАВИТЬ запись",
          "2 -> ВЫВЕСТИ все записи",
          "3 -> ПРОЧИТАТЬ запись",
          "4 -> РЕДАКТИРОВАТЬ запись",
          "5 -> УДАЛИТЬ запись",
          "0 -> ВЫХОД",
          "-------------------------",
          sep='\n')
    command = input("Введите № меню: -> ")
    commander = Command(file_path)
    if command == '1':  # ДОБАВЛЕНИЕ новой записи
        commander.add()
    elif command == '5':  # УДАЛЕНИЕ записи из JSON
        commander.delete()
    elif command == '4':  # РЕДАКТИРОВАНИЕ записи
        commander.edit()
    elif command == '2':  # ЧТЕНИЕ всех сохраненных записей из JSON
        commander.read_all()
    elif command == '3':  # ЧТЕНИЕ сохраненных записей из JSON
        commander.read_nom()
    elif command == '0':  # ВЫХОД из программы
        break
    else:
        print("-------------------------\n",
              "Вы ввели неверную команду!")
