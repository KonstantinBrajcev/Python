from Manager import NoteManager

manager = NoteManager()


class Command:

    def __init__(self, file_path):  # ИНИЦИАЛИЗАЦИЯ запросов команд
        self.file_path = file_path

    def add(self):  # ДОБАВЛЕНИЕ новой записи
        title = input("Введите заголовок заметки: ")
        body = input("Введите тело заметки: ")
        manager.add_note(title, body, self.file_path)

    def delete(self):  # УДАЛЕНИЕ записи из JSON
        id_to_del = input(f"Введите ID удаляемой записи: ")
        manager.del_from_json(id_to_del, self.file_path)

    def edit(self):  # РЕДАКТИРОВАНИЕ записи
        id_to_edit = input(f"Введите ID редактируемой записи: ")
        if manager.check_existence(id_to_edit, self.file_path):
            new_title = input("Новый заголовок: ")
            new_body = input("Новая заметка: ")
            manager.edit_from_json(id_to_edit, new_title,
                                   new_body, self.file_path)
        else:
            NoteManager.print_none(self, id_to_edit)

    def read_all(self):  # ЧТЕНИЕ сохраненных записей из JSON
        manager.read_all_from_json(self.file_path)

    def read_nom(self):  # ЧТЕНИЕ сохраненных записей из JSON
        nom_id = input("Введи ID записи: ")
        manager.read_nom_from_json(nom_id, self.file_path)
