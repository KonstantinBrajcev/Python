from random import choice

word_list = [
    'год', 'человек', 'время', 'дело', 'жизнь', 'день', 'рука', 'работа', 'слово', 'место',
    'вопрос', 'лицо', 'глаз', 'страна', 'друг', 'сторона', 'дом', 'случай', 'ребенок', 'голова',
    'система', 'вид', 'конец', 'отношение', 'город', 'часть', 'женщина', 'проблема', 'земля',
    'решение', 'власть', 'машина', 'закон', 'час', 'образ', 'отец', 'история', 'нога', 'вода',
    'война', 'возможность', 'компания', 'результат', 'дверь', 'народ', 'область', 'число',
    'голос', 'развитие', 'группа', 'жена', 'процесс', 'условие', 'книга', 'ночь', 'суд', 'деньга',
    'уровень', 'начало', 'государство', 'стол', 'средство', 'связь', 'имя', 'президент', 'форма',
    'путь', 'организация', 'качество', 'действие', 'статья', 'общество', 'ситуация', 'деятельность',
    'школа', 'душа', 'дорога', 'язык', 'взгляд', 'момент', 'минута', 'месяц', 'порядок', 'цель',
    'программа', 'муж', 'помощь', 'мысль', 'вечер', 'орган', 'правительство', 'рынок', 'предприятие',
    'партия', 'роль', 'смысл', 'мама', 'мера', 'улица', 'состояние', 'задача', 'информация', 'театр',
    'внимание', 'производство', 'квартира', 'труд', 'тело', 'письмо', 'центр', 'утро', 'мать', 'комната',
    'семья', 'сын', 'смерть', 'положение', 'интерес', 'федерация', 'век', 'идея', 'управление', 'автор',
    'окно', 'ответ', 'совет', 'разговор', 'мужчина', 'ряд', 'счет', 'мнение', 'цена', 'точка', 'план',
    'проект', 'глава', 'материал', 'основа', 'причина', 'движение', 'культура', 'сердце', 'рубль', 'наука',
    'документ', 'неделя', 'вещь', 'чувство', 'правило', 'служба', 'газета', 'срок', 'институт', 'ход',
    'стена', 'директор', 'плечо', 'опыт', 'встреча', 'принцип', 'событие', 'структура', 'количество', 'товарищ',
    'создание', 'значение', 'объект', 'гражданин', 'очередь', 'период', 'образование', 'состав', 'пример',
    'лес', 'исследование', 'девушка', 'данные', 'палец', 'судьба', 'тип', 'метод', 'политика', 'армия', 'брат',
    'представитель', 'борьба', 'использование', 'шаг', 'игра', 'участие', 'территория', 'край', 'размер', 'номер',
    'район', 'население', 'банк', 'начальник', 'класс', 'зал', 'изменение', 'большинство', 'характер', 'кровь',
    'направление', 'позиция', 'герой', 'течение', 'девочка', 'искусство', 'гость', 'воздух', 'мальчик', 'фильм',
    'договор', 'регион', 'выбор', 'свобода', 'врач', 'экономика', 'небо', 'факт', 'церковь', 'завод', 'фирма',
    'бизнес', 'союз', 'деньги', 'специалист', 'род', 'команда', 'руководитель', 'спина', 'дух', 'музыка',
    'способ', 'хозяин', 'поле', 'доллар', 'память', 'природа', 'дерево', 'оценка', 'объем', 'картина',
    'процент', 'требование', 'писатель', 'сцена', 'анализ', 'основание', 'повод', 'вариант', 'берег',
    'модель', 'степень', 'самолет', 'телефон', 'граница', 'песня', 'половина', 'министр', 'угол', 'зрение',
    'предмет', 'литература', 'операция', 'двор', 'спектакль', 'руководство', 'солнце', 'автомобиль', 'родитель',
    'участник', 'журнал', 'база', 'пространство', 'защита', 'название', 'стих', 'море', 'удар', 'знание',
    'солдат', 'миллион', 'строительство', 'технология', 'председатель', 'сон', 'сознание', 'бумага', 'реформа',
    'оружие', 'линия', 'текст', 'выход', 'ребята', 'магазин', 'соответствие', 'участок', 'услуга', 'поэт',
    'предложение', 'желание', 'пара', 'успех', 'среда', 'возраст', 'комплекс', 'бюджет', 'представление',
    'площадь', 'генерал', 'господин', 'дочь', 'понятие', 'кабинет', 'безопасность', 'фонд', 'сфера', 'папа',
    'сотрудник', 'продукция', 'будущее', 'продукт', 'содержание', 'художник', 'республика', 'сумма', 'контроль',
    'парень', 'ветер', 'хозяйство', 'помочь', 'курс', 'губа', 'река', 'грудь', 'огонь', 'нос', 'волос', 'ухо',
    'отсутствие', 'радость', 'сад', 'подготовка', 'необходимость', 'доктор', 'лето', 'камень', 'здание',
    'капитан', 'собака', 'итог', 'рис', 'техника', 'элемент', 'источник', 'деревня', 'депутат', 'проведение',
    'рот', 'масса', 'комиссия', 'цвет', 'рассказ', 'функция', 'определение', 'мужик', 'обеспечение',
    'обстоятельство', 'работник', 'разработка', 'лист', 'звезда', 'гора', 'применение', 'победа', 'товар',
    'воля', 'зона', 'предел', 'целое', 'личность', 'офицер', 'влияние', 'поддержка', 'ответственность',
]


def get_word():
    return choice(word_list).upper()


# функция получения текущего состояния
def display_hangman(tries):
    stages = [
        '''
                \|||/ 
                (o o)
        ----ooO--(_)---------
       |                     |
       |      К О Н Е Ц      |
       |                     |
       '---------------ooO---'
               |__|__| 
               /_'Y'_\  
              (__/ \__)  
                ''',
        '''
                \|||/ 
                (o o)
        ----ooO--(_)---------
       |                     |
       |  Осталось 1 попытка |
       |                     |
       '---------------ooO---'
               |__|__| 
               /_'Y'_\  
              (__/ \__)  
                ''',
        '''
                \|||/ 
                (o o)
        ----ooO--(_)---------
       |                     |
       |  Осталось 2 попытки |
       |                     |
       '---------------ooO---'
               |__|__| 
               /_'Y'_\  
              (__/ \__)  
                ''',

        '''
                \|||/ 
                (o o)
        ----ooO--(_)---------
       |                     |
       |  Осталось 3 попытки |
       |                     |
       '---------------ooO---'
               |__|__| 
               /_'Y'_\  
              (__/ \__)  
                ''',
        '''
                \|||/ 
                (o o)
        ----ooO--(_)---------
       |                     |
       |  Осталось 4 попытки |
       |                     |
       '---------------ooO---'
               |__|__| 
               /_'Y'_\  
              (__/ \__)  
                ''',
        '''
                \|||/ 
                (o o)
        ----ooO--(_)---------
       |                     |
       |  Осталось 5 попыток |
       |                     |
       '---------------ooO---'
               |__|__| 
               /_'Y'_\  
              (__/ \__)  
                ''',
        '''
                \|||/
                (o o)
        ----ooO--(_)---------
       |                     |
       |  Осталось 6 попыток |
       |                     |
       '---------------ooO---'
               |__|__| 
               /_'Y'_\  
              (__/ \__)  
              '''
    ]
    return stages[tries]


def letter_finder(ltr, original_word, hidden_word):
    lst_index = []
    for i, letter in enumerate(original_word):
        if letter == ltr:
            lst_index.append(i)
    for i in lst_index:
        hidden_word[i] = ltr
    return hidden_word


def play(word):
    end = '- ' * 30
    pc_word = list(word)
    # строка, содержащая символы _ на каждую букву задуманного слова
    word_completion = list('_' * len(word))
    guessed_letters = []               # список уже названных букв
    guessed_words = []                 # список уже названных слов
    tries = 6                          # количество попыток
    print('\nДавайте играть в угадайку слов!')
    print('Загаданное слово состоит из', len(pc_word), 'букв')
    # print(pc_word)  # Выводит загаданное слово
    while True:
        easy_or_not = input(
            'Чтобы было проще, хотите открыть первую и последнюю буквы? (да/нет) ')
        if easy_or_not.isalpha():
            if easy_or_not.lower() == 'да':
                word_completion[0] = pc_word[0]
                word_completion[-1] = pc_word[-1]
                break
            elif easy_or_not.lower() == 'нет':
                print('Ваш выбор, играем! =)')
                break
            else:
                print('Не понял, повторите...')
        else:
            print('Не понял, повторите...')

    while tries:
        if word_completion == pc_word:  # Проверка на случай угадывания слова по одной букве
            print('Ура! Вы выиграли!')
            break
        print(display_hangman(tries))
        print(end)
        print(*word_completion)
        user_letter = input('\nВведите букву или слово полностью: ').upper()
        if len(user_letter) == len(pc_word):  # Проверка на слово полностью
            if user_letter in guessed_words:
                print('Такое слово вы уже пробовали, это не оно!')
                continue
            guessed_words.append(user_letter.upper())
            if user_letter.upper() == ''.join(pc_word):  # Проверка на ввод слова полностью
                print('Ура! Вы выиграли!')
                break
        if not user_letter.isalpha():
            print('Вы что-то неправильно ввели!')
            continue
        elif user_letter in guessed_letters:
            print('Вы уже называли такую букву!')
            continue
        # Добавляем букву в список уже названных букв
        guessed_letters.append(user_letter)
        if user_letter in pc_word:  # Если буква была угадана
            letter_finder(user_letter, pc_word, word_completion)
            tries -= 1
        else:
            print('Вы не угадали букву/слово!')
            tries -= 1
    print('Игра закончена!')
    print(display_hangman(0))
    print('Загаданное слово было:', *pc_word)
    return end


print(play(get_word()))


while True:
    repeat = input('Хотите сыграть ещё раз? (да/нет) ')
    if repeat.isalpha():
        if repeat.lower() == 'да':
            play(get_word())
        elif repeat.lower() == 'нет':
            print('Спасибо за игру! До встречи! =)', '* ' * 30, sep='\n')
            break
        else:
            print('Не понял, повторите')
    else:
        print('Не понял, повторите')
