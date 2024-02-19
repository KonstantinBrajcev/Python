import Print
import datetime
from Note import Note
from FileManager import FileManager


class NoteManager:

    def __init__(self):  # ИНИЦИАЛИЗАЦИЯ выполнения команд
        self.notes = []

    def read_all_from_json(self, file_path):  # Читаем записи из JSON
        data = FileManager.read_data(file_path)
        self.notes = [Note(**note) for note in data]
        Print.print_all(self, self.notes)

    def add_note(self, title, body, file_path):  # Добавляем записи в JSON
        data = FileManager.read_data(file_path)
        max_id = max([note['note_id'] for note in data], default=0)  # Max ID
        max_id += 1
        new_note = Note(max_id, title, body,
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.notes.append(new_note)  # Добавляем новую запись
        data = FileManager.read_data(file_path)
        data.append(new_note.__dict__)  # Добавляем новую запись к загруженным
        FileManager.write_data(data, file_path)
        Print.print_new(self, max_id, new_note, file_path)

    def read_nom_from_json(self, nom_id, file_path):  # Читаем записи из JSON
        nom_id = int(nom_id)
        if self.check_existence(nom_id, file_path):
            data = FileManager.read_data(file_path)
            note_found = False
            self.notes = [Note(**note) for note in data]
            for note in self.notes:
                if note.note_id == nom_id:
                    Print.print_nom(self, note)
                    note_found = True
            if not note_found:
                Print.print_none(self, nom_id)
        else:
            Print.print_none(self, nom_id)

    def del_from_json(self, id_to_del, file_path):
        id_to_del = int(id_to_del)  # Преобразование в целое число
        if self.check_existence(id_to_del, file_path):
            data = FileManager.read_data(file_path)
            data = [note for note in data if note['note_id'] != id_to_del]
            FileManager.write_data(data, file_path)
            Print.print_del(self, id_to_del)
        else:
            Print.print_none(self, id_to_del)

    def edit_from_json(self, id_to_edit, new_title, new_body, file_path):
        id_to_edit = int(id_to_edit)  # Преобразование в целое число
        data = FileManager.read_data(file_path)
        edit_note = None
        for note in data:
            if note['note_id'] == id_to_edit:
                note['title'] = new_title
                note['body'] = new_body
                edit_note = Note(
                    note['note_id'], note['title'], note['body'], note['timestamp'])
                break
        FileManager.write_data(data, file_path)
        Print.print_edit(self, edit_note)

    def check_existence(self, id_to_edit, file_path):
        data = FileManager.read_data(file_path)
        for note in data:
            if note['note_id'] == int(id_to_edit):
                return True
        return False
