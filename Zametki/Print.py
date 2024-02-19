def print_none(self, nom):
    print("-------------------------\n",
          f"Запись № {nom} не найдена.")


def print_all(self, notes):
    print("-------------------------")
    for note in notes:
        print(str(note))


def print_edit(self, edit_note):
    print("-------------------------\n",
          f"Обновленная строка: {str(edit_note)}")


def print_new(self, max_id, new_note, file_path):
    print("-------------------------\n",
          f"Заметка № {max_id} записана в файл {file_path}.\n"
          f"{str(new_note)}")


def print_nom(self, note):
    print("-------------------------\n",
          f"{str(note)}")


def print_del(self, id_to_del):
    print("-------------------------\n",
          f"Запись № {id_to_del} удалена.")
