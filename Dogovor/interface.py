import sys
import os
import tkinter as tk
import datetime
from tkinter import IntVar
import docx
import tkinter
from tokenize import String
from tkinter import StringVar
from babel import numbers
from tkcalendar import DateEntry
from tkinter import filedialog
from replase import replace_text

# Объявление глобальных переменных
global Label7
global Label8


def create_interface(file_path, file_name):
    # -------------------------------------------------------------------
    # ------------------СОЗДАЕМ ИНТЕРФЕЙС ПРОГРАММЫ----------------------
    # -------------------------------------------------------------------
    global sel_value
    global label7
    global label8
    root = tk.Tk()
    root.withdraw()  # Скрыть основное окно
    popup = tk.Toplevel()
    popup.title("Реквизиты")
    sel_value = StringVar()  # Переменная для хранения выбранного значения
    current_date = datetime.date.today()  # Переменная для хранения текущей даты

    # -------------------------------------------------------------------
    # -----------------------СОЗДАНИЕ ФРЭЙМОВ----------------------------
    # -------------------------------------------------------------------
    frame_file = tk.LabelFrame(popup, text="Файл")
    frame_file.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
    frame_dog = tk.LabelFrame(popup, text="Договор")
    frame_dog.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
    frame_org = tk.LabelFrame(popup, text="Организация")
    frame_org.grid(row=2, column=0, padx=10, pady=5, sticky='ew')
    frame_bank = tk.LabelFrame(popup, text="Банк")
    frame_bank.grid(row=3, column=0, padx=10, pady=5, sticky='ew')
    frame_button = tk.LabelFrame(popup, text="Состояние")
    frame_button.grid(row=5, column=0, padx=10, pady=5, sticky='ew')
    frame_right = tk.LabelFrame(popup)
    frame_right.grid(row=0, column=1, rowspan=6, padx=10, pady=5, sticky='ns')

    # -------------------------------------------------------------------
    # ------------------------ФРЭЙМ с КНОПКАМИ---------------------------
    # -------------------------------------------------------------------
    frame_right.grid_rowconfigure(1, weight=1)  # 2я строка растягивается
    # ------------------------КНОПКА СОХРАНИТЬ---------------------------
    submit_button = tk.Button(frame_right, text="Сформировать", command=lambda:
                              replace_text(file_path,  # Адрес файла
                                           _cal_.get(),  # Календарь
                                           _nomDog_.get(),  # Номер договора
                                           _predmetDog_.get(),  # Предмет договора
                                           _adressDog_.get(),  # Адрес подъемника
                                           _fin_.get(),  # Финансирование
                                           _cost_.get(),  # Сумма по договору
                                           sel_value.get(),  # Тип договора
                                           _unp_.get(),  # УНП
                                           _okpo_.get(),  # ОКПО
                                           _nameOrg_.get(),  # Наименование организации
                                           _adresOrg_.get(),  # Адрес Организации
                                           _nameOrgSokr_.get(),  # Сокращенное Название Организации
                                           _fioDir_.get(),  # ФИО Директора
                                           _doljnost_.get(),  # Должность руководителя
                                           _osnovanie_.get(),  # Основание
                                           _email_.get(),  # Электронная почта
                                           _phone_.get(),  # Телефон
                                           _nameBank_.get(),  # Наименование Банка
                                           _adresBank_.get(),  # Адрес Банка
                                           _iban_.get(),  # IBAN
                                           _swift_.get()  # SWIFT
                                           ), width=15, height=2)
    submit_button.grid(row=1, column=0, padx=5, pady=5, sticky="s")
    # -----------------------КНОПКА ВЫХОД------------------------------
    exit_button = tk.Button(frame_right, text="Выход",
                            command=lambda: sys.exit(), width=15, height=2)
    exit_button.grid(row=2, column=0, padx=5, pady=5, sticky="s")

    # -------------------------------------------------------------------
    # -----------------------ФРЭЙМ ДОГОВОР-------------------------------
    # -------------------------------------------------------------------
    frame_dog.grid_columnconfigure(0, minsize=130)
    frame_dog.grid_columnconfigure(1, minsize=180)
    frame_dog.grid_columnconfigure(2, minsize=110)
    frame_dog.grid_columnconfigure(3, minsize=120)
    frame_dog.grid_columnconfigure(1, weight=1)  # 2й столбец растягивается
    frame_dog.grid_columnconfigure(2, weight=1)  # 3й столбец растягивается
    frame_dog.grid_columnconfigure(3, weight=1)  # 4й столбец растягивается
    # ---------Номер договора
    label1 = tk.Label(frame_dog, text="Номер договора:")
    label1.grid(row=0, column=2, padx=10, pady=5, sticky="e")
    # ---------Ввод номера договора
    _nomDog_ = tk.Entry(frame_dog)
    _nomDog_.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
    _nomDog_.bind("<KeyRelease>", lambda event: show_selected(_nomDog_, _cal_))

    # ---------Дата договора
    label3 = tk.Label(frame_dog, text="Дата договора:")
    label3.grid(row=0, column=0, padx=10, pady=5, sticky="e")
    # ---------Календарь
    _cal_ = DateEntry(frame_dog, width=12, year=current_date.year, month=current_date.month, day=current_date.day,
                      background='darkblue', foreground='white', borderwidth=1)
    _cal_.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
    _cal_.bind("<<DateEntrySelected>>",
               lambda event: show_selected(_nomDog_, _cal_))

    # ---------Предмет договора
    label6 = tk.Label(frame_dog, text="Предмет договора/подъемник:")
    label6.grid(row=1, column=0, padx=10, pady=5, sticky="e")
    # ---------Ввод предмета договора
    _predmetDog_ = tk.Entry(frame_dog)
    _predmetDog_.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

    # ---------Адрес договора
    label23 = tk.Label(frame_dog, text="Адрес оборудования:")
    label23.grid(row=1, column=2, padx=10, pady=5, sticky="e")
    # ---------Ввод предмета договора
    _adressDog_ = tk.Entry(frame_dog)
    _adressDog_.grid(row=1, column=3, padx=10, pady=5, sticky="ew")

    # ---------Финансирование------------------
    label20 = tk.Label(frame_dog, text="Финансирование:")
    label20.grid(row=2, column=0, padx=10, pady=5, sticky="e")
    # ---------Ввод финансирования------
    _fin_ = tk.Entry(frame_dog)
    _fin_.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

    # ---------Сумма договора------------------
    label22 = tk.Label(frame_dog, text="Сумма:")
    label22.grid(row=2, column=2, padx=10, pady=5, sticky="e")
    # ---------Ввод суммы договора------
    _cost_ = tk.Entry(frame_dog)
    _cost_.grid(row=2, column=3, padx=10, pady=5, sticky="ew")

    # ----------Выбор ТИПА договора----------------
    label21 = tk.Label(frame_dog, text="Вид договора:")
    label21.grid(row=3, column=0, padx=10, pady=5, sticky="e")
    # ---------Чекбокс Обслуживание----------------
    radiobutton_1 = tk.Radiobutton(
        frame_dog, text="Долгосрочный", variable=sel_value, value="долгосрочный", command=lambda: show_selected(_nomDog_, _cal_))
    radiobutton_1.grid(row=3, column=1, padx=10, pady=5, sticky="e")
    # ---------Чекбокс Ремонт----------------------
    radiobutton_2 = tk.Radiobutton(
        frame_dog, text="Разовый", variable=sel_value, value="разовый", command=lambda: show_selected(_nomDog_, _cal_))
    radiobutton_2.grid(row=3, column=1, padx=10, pady=5, sticky="w")
    sel_value.set("разовый")

    # -------------------------------------------------------------------
    # ----------------------------ФРЭЙМ ОРГАНИЗАЦИЯ----------------------
    # -------------------------------------------------------------------
    frame_org.grid_columnconfigure(0, minsize=160)
    frame_org.grid_columnconfigure(2, minsize=100)
    frame_org.grid_columnconfigure(1, weight=1)  # 2й столбец растягивается
    frame_org.grid_columnconfigure(2, weight=1)  # 3й столбец растягивается
    frame_org.grid_columnconfigure(3, weight=1)  # 4й столбец растягивается

    # ---------УНП организации-----------
    label13 = tk.Label(frame_org, text="УНП:")
    label13.grid(row=0, column=0, padx=10, pady=5, sticky="e")
    # ---------Форма наименование организации
    _unp_ = tk.Entry(frame_org)
    _unp_.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

    # ---------ОКПО организации-----------
    label14 = tk.Label(frame_org, text="ОКПО:")
    label14.grid(row=0, column=2, padx=10, pady=5, sticky="e")
    # ---------Форма наименование организации
    _okpo_ = tk.Entry(frame_org)
    _okpo_.grid(row=0, column=3, padx=10, pady=5, sticky="ew")

    # ---------Наименование организации-----------
    label2 = tk.Label(frame_org, text="Полное название:")
    label2.grid(row=1, column=0, padx=10, pady=5, sticky="e")
    # ---------Форма наименование организации
    _nameOrg_ = tk.Entry(frame_org)
    _nameOrg_.grid(row=1, column=1, columnspan=3, padx=10, pady=5, sticky="ew")

    # ---------Сокращенное наименование-----------
    label16 = tk.Label(frame_org, text="Сокращенное название:")
    label16.grid(row=3, column=0, padx=10, pady=5, sticky="e")
    # ---------Форма наименование организации
    _nameOrgSokr_ = tk.Entry(frame_org)
    _nameOrgSokr_.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

    # ---------Руководитель организации
    label4 = tk.Label(frame_org, text="Фамилия И.О.:")
    label4.grid(row=3, column=2, padx=10, pady=5, sticky="e")
    # ---------Форма ввода руководителя организации
    _fioDir_ = tk.Entry(frame_org)
    _fioDir_.grid(row=3, column=3, padx=10, pady=5, sticky="ew")

    # ---------Адрес организации
    label3 = tk.Label(frame_org, text="Адрес организации:")
    label3.grid(row=2, column=0, padx=10, pady=5, sticky="e")
    # ---------Форма ввода адреса организации
    _adresOrg_ = tk.Entry(frame_org)
    _adresOrg_.grid(row=2, column=1, columnspan=3,
                    padx=10, pady=5, sticky="ew")

    # ---------Руководитель организации
    label15 = tk.Label(frame_org, text="Должность:")
    label15.grid(row=4, column=0, padx=10, pady=5, sticky="e")
    # ---------Ввод руководителя организации
    _doljnost_ = tk.Entry(frame_org)
    _doljnost_.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

    # ---------Действует на основании
    label5 = tk.Label(frame_org, text="На основании:")
    label5.grid(row=4, column=2, padx=10, pady=5, sticky="e")
    # ---------Ввода даты и номера документа
    _osnovanie_ = tk.Entry(frame_org)
    _osnovanie_.grid(row=4, column=3, padx=10, pady=5, sticky="ew")

    # ---------Электронная почта
    label17 = tk.Label(frame_org, text="Электронная почта:")
    label17.grid(row=5, column=0, padx=10, pady=5, sticky="e")
    # ---------Ввод электронной почты
    _email_ = tk.Entry(frame_org, width=70)
    _email_.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

    # ---------Контактный телефон
    label18 = tk.Label(frame_org, text="Телефон:")
    label18.grid(row=5, column=2, padx=10, pady=5, sticky="e")
    # ---------Ввод телефона
    _phone_ = tk.Entry(frame_org, width=70)
    _phone_.grid(row=5, column=3, padx=10, pady=5, sticky="ew")

    # -------------------------------------------------------------------
    # --------------------------ФРЭЙМ БАНК-------------------------------
    # -------------------------------------------------------------------
    frame_bank.grid_columnconfigure(1, weight=1)  # 2й столбец растягивается
    # ---------Наимеование банка-----------
    label9 = tk.Label(frame_bank, text='Название банка:')
    label9.grid(row=0, column=0, padx=10, pady=5, sticky="e")
    # ---------Ввод наименования банка
    _nameBank_ = tk.Entry(frame_bank)
    _nameBank_.grid(row=0, column=1, columnspan=3,
                    padx=10, pady=5, sticky="ew")

    # ---------Адрес банка
    label10 = tk.Label(frame_bank, text="Адрес банка:")
    label10.grid(row=1, column=0, padx=10, pady=5, sticky="e")
    # ---------Ввод адреса банка
    _adresBank_ = tk.Entry(frame_bank, width=70)
    _adresBank_.grid(row=1, column=1, columnspan=3,
                     padx=10, pady=5, sticky="ew")

    # ---------Расчетный счет организации
    label11 = tk.Label(frame_bank, text="Расчетный счет:")
    label11.grid(row=2, column=0, padx=10, pady=5, sticky="e")
    # ---------Ввод расчетного счета
    _iban_ = tk.Entry(frame_bank, width=70)
    _iban_.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

    # ---------Код Банка
    label12 = tk.Label(frame_bank, text="Код банка:")
    label12.grid(row=2, column=2, padx=10, pady=5, sticky="e")
    # ---------Ввод кода банка
    _swift_ = tk.Entry(frame_bank, width=70)
    _swift_.grid(row=2, column=3, padx=10, pady=5, sticky="ew")

    # -----------------------------------------------------------
    # ----------------------ФРЭЙМ ФАЙЛ--------------------------
    # -----------------------------------------------------------
    frame_file.grid_columnconfigure(0, weight=1)  # 1я строка растягивается
    # ---------Индикация открытия файла-----------
    if os.path.isfile(file_path):
        label8 = tk.Label(frame_file, text=f"Файл открыт: {file_path}")
    else:
        label8 = tk.Label(
            frame_file, text=f"Файл не открыт! Откройте файл '{file_name}'")
    label8.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    # ---------Индикация формируемого договора------
    label7 = tk.Label(
        frame_file, text=f"Будет сформирован {sel_value.get()} договор № {_nomDog_.get()} от {_cal_.get_date().strftime('%d.%m.%Y')} года.")
    label7.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    # ---------Кнопка Открытия файла----------
    open_button = tk.Button(frame_file, text="Открыть",
                            command=open_file, width=15, height=3)
    open_button.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="e")

    # -------------------------------------------------------------------
    popup.update_idletasks()  # Обновление размеров и расположения виджетов
    popup.columnconfigure(0, weight=1)  # Растягивание по горизонтали
    frame_dog.columnconfigure(0, weight=1)  # Растягивание по горизонтали
    frame_org.columnconfigure(0, weight=1)  # Растягивание по горизонтали
    frame_button.columnconfigure(0, weight=1)  # Растягивание по горизонтали
    # center_window(root)
    popup.state("zoomed")
    # center_window(popup)
    root.mainloop()
    # -------------------------------------------------------------------


# def center_window(window):
#     window.update_idletasks()
#     width = window.winfo_width()
#     height = window.winfo_height()
#     screen_width = window.winfo_screenwidth()
#     screen_height = window.winfo_screenheight()
#     x = (screen_width - width) // 2
#     y = (screen_height - height) // 2
#     window.geometry('+{}+{}'.format(x, y))


def submit_values(file_path, _nomDog_, _nameOrg_, _cal_, _predmetDog_, _fioDirector_, _osnovanie_, _adresOrg_):
    # Нажимаем кнопку СФОРМИРОВАТЬ
    nomDog = _nomDog_.get()     # Получаем номер договора
    nameOrg = _nameOrg_.get()   # Получаем наименование организации
    cal = _cal_.get()           # Получаем ДАТУ
    predmetDog = _predmetDog_.get()  # Получаем предмет договора
    fioDirector = _fioDirector_.get()  # Получаем ФИО Директора
    osnovanie = _osnovanie_.get()  # Получаем основание
    adresOrg = _adresOrg_.get()  # Получаем адрес организации
    # Пример использования функции
    replace_text(file_path, nomDog, nameOrg, cal,
                 predmetDog, fioDirector, osnovanie, adresOrg)


def open_file():  # При открытии файла
    file_path = filedialog.askopenfilename(
        filetypes=[("Word files", "*.docx"), ("All files", "*.*")])
    if file_path:
        label8.config(text=f"Файл открыт: {file_path}")


def show_selected(_nomDog_, _cal_):  # При смене радиобуттона
    label7.config(
        text=f"Будет сформирован {sel_value.get()} договор № {_nomDog_.get()} от {_cal_.get_date().strftime('%d.%m.%Y')}")
