from interface import create_interface
from babel import numbers
import docx
import os

if __name__ == "__main__":
    current_directory = os.getcwd()  # Получаем текущую директорию
    file_name = "dogovor_new.docx"
    # Путь к файлу "dogovor.docx" в текущей директории
    file_path = os.path.join(current_directory, file_name)
    create_interface(file_path, file_name)

# pyinstaller --onefile main.py
# pyinstaller --onefile --noconsole main.py
# export PATH=$PATH:/c/Users/User/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/Scripts
# export PATH=$PATH:/c/Users/User/AppData/Local/Programs/Python/Python312/Scripts/
