from models import User
from flask import jsonify
from flask import Flask


# Класс {Poduct} создает сущность продукта с атрибутами {name}, {owner} и {users}:
# {name} - название продукта
# {owner} - владелец
# {users} - список пользователей с доступом к продукту.
# Методы {add_user} и {remove_user} позволяют добавлять и удалять пользователей из списка соответственно.
# Метод {add_lesson} позволяет добавить уроки.
class Product:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner
        self.users = []
        self.lessons = []

    def add_user(self, user):
        self.users.append(user)

    def remove_user(self, user):
        self.users.remove(user)

    def add_lesson(self, lesson):
        self.lessons.append(lesson)

# Класс {Lesson} создает сущность урока с атрибутами:
# {name} - название урока
# {video_link} - ссылка на видео
# {duration} - длительность просмотра
# Методы {add_product} и {remove_product} позволяют добавлять и удалять уроки из списка продуктов, в которых они находятся.
# Метод {add_user_lesson} позволяет добавить уроки пользователю.


class Lesson:
    def __init__(self, name, video_link, duration):
        self.name = name
        self.user_lessons = []
        self.video_link = video_link
        self.duration = duration
        self.products = []

    def add_product(self, product):
        self.products.append(product)

    def remove_product(self, product):
        self.products.remove(product)

    def add_user_lesson(self, user_lesson):
        self.user_lessons.append(user_lesson)

# Класс {UserLesson} создает сущность для каждого пользователя урока с атрибутами:
# {user} - пользователь
# {lesson} - урок
# {view_time} - время просмотра
# {status} - статус.
# Метод {update_view_time} позволяет обновлять время просмотра и соответственно изменять статус на
# "Просмотрено" или "Не просмотрено" в зависимости от того, просмотрел ли пользователь 80% ролика.


class UserLesson:
    def __init__(self, user, lesson):
        self.user = user
        self.lesson = lesson
        self.view_time = 0
        self.status = "Не просмотрено"

    def set_status(self, status):
        self.status = status

    def set_view_time(self, view_time):
        self.view_time = view_time

    def update_view_time(self, time):
        self.view_time = time

        if self.view_time >= self.lesson.duration * 0.8:
            self.status = "Просмотрено"
        else:
            self.status = "Не просмотрено"

# --------------------NEW-------------------------------
# Класс {Platform} создает продукты, уроки и пользователей


class Platform:
    def __init__(self):
        self.products = []
        self.users = []

    def create_product(self, name, owner):
        product = Product(name, owner)
        self.products.append(product)
        return product

    def create_lesson(self, name, video_url, duration):
        lesson = Lesson(name, video_url, duration)
        return lesson

    def create_user(self, name):
        user = User(name)
        self.users.append(user)
        return user


class User:
    def __init__(self, name):
        self.name = name


# ------------------------------------------------------
# реализация API для выведения списка всех уроков по всем продуктам к которым пользователь имеет доступ используем фрэймворк Flask
app = Flask(__name__)

# # Создаем экземпляры продуктов, уроков и пользователей
# product1 = Product("Продукт 1", "Владелец 1")
# product2 = Product("Продукт 2", "Владелец 2")

# lesson1 = Lesson("Урок 1", "https://www.youtube.com/watch?v=abc123", 300)
# lesson2 = Lesson("Урок 2", "https://www.youtube.com/watch?v=def456", 600)

# user1 = User("Пользователь 1")
# user2 = User("Пользователь 2")

# # Добавляем пользователей в продукты
# product1.add_user(user1)
# product1.add_user(user2)
# product2.add_user(user2)

# # Добавляем уроки в продукты
# product1.add_lesson(lesson1)
# product2.add_lesson(lesson1)
# product2.add_lesson(lesson2)

# # Добавляем пользователей в уроки
# user_lesson1 = UserLesson(user1, lesson1)
# user_lesson2 = UserLesson(user2, lesson1)
# user_lesson3 = UserLesson(user2, lesson2)

# lesson1.add_user_lesson(user_lesson1)
# lesson1.add_user_lesson(user_lesson2)
# lesson2.add_user_lesson(user_lesson3)

# Создаем эндпоинт для получения списка всех уроков и информации о статусе и времени просмотра


@app.route('/lessons/<username>')
def get_lessons(username):
    user_lessons = []
    for product in Product.products:
        if username in [user.name for user in product.users]:
            for lesson in product.lessons:
                user_lesson = lesson.get_user_lesson(username)
                user_lessons.append({
                    "Имя продукта": product.name,
                    "Имя урока": lesson.name,
                    "Статус": user_lesson.status,
                    "Просмотренное время": user_lesson.view_time
                })
    return jsonify(user_lessons)


if __name__ == '__main__':
    app.run()


# -------------------------------------------------------------
# реализация API с выведением списка уроков по конкретному продукту к которому пользователь имеет доступ,
# с выведением информации о статусе и времени просмотра, а также датой последнего просмотра ролика


# Создаем эндпоинт для получения списка уроков по конкретному продукту и информации о статусе и времени просмотра


@app.route('/lessons/<username>/<product_name>')
def get_lessons_by_product(username, product_name):
    user_lessons = []
    for product in Product.products:
        if product.name == product_name and username in [user.name for user in product.users]:
            for lesson in product.lessons:
                user_lesson = lesson.get_user_lesson(username)
                if user_lesson:
                    user_lessons.append({
                        "название урока": lesson.name,
                        "ссылка видео": lesson.video_link,
                        "продолжительность": lesson.duration,
                        "статус": user_lesson.status,
                        "просмотренное время": user_lesson.view_time,
                        "последний просмотр": user_lesson.last_viewed
                    })
            return jsonify({"Урок": user_lessons})
    return jsonify({"Сообщение": "Продукт не найден или не имеет доступа к этому продукту"})


# -------------------------------------------------------------------

app = Flask(__name__)

# # Создаем экземпляры продуктов, уроков и пользователей
# product1 = Product("Продукт 1", "Владелец 1")
# product2 = Product("Продукт 2", "Владелец 2")

# lesson1 = Lesson("Урок 1", "https://www.youtube.com/watch?v=abc123", 300)
# lesson2 = Lesson("Урок 2", "https://www.youtube.com/watch?v=def456", 600)

# user1 = User("Пользователь 1")
# user2 = User("Пользователь 2")

# # Добавляем пользователей в продукты
# product1.add_user(user1)
# product1.add_user(user2)
# product2.add_user(user2)

# # Добавляем уроки в продукты
# product1.add_lesson(lesson1)
# product2.add_lesson(lesson1)
# product2.add_lesson(lesson2)

# # Добавляем пользователей в уроки
# user_lesson1 = UserLesson(user1, lesson1)
# user_lesson2 = UserLesson(user2, lesson1)
# user_lesson3 = UserLesson(user2, lesson2)

# lesson1.add_user_lesson(user_lesson1)
# lesson1.add_user_lesson(user_lesson2)
# lesson2.add_user_lesson(user_lesson3)

# Создаем эндпоинт для получения статистики по продуктам


@app.route('/stats')
def get_stats():
    stats = []
    for product in Product.products:
        users_count = len(product.users)
        lessons_count = len(product.lessons)
        total_viewed_lessons = 0
        total_view_time = 0

        for lesson in product.lessons:
            for user_lesson in lesson.user_lessons:
                if user_lesson.status == "Просмотрено":
                    total_viewed_lessons += 1
                    total_view_time += user_lesson.view_time

        percentage_of_access = (users_count / User.users_count) * 100

        stats.append({
            "имя продукта": product.name,
            "счетчик пользователей": users_count,
            "Счетчик уроков": lessons_count,
            "всего просмотрено уроков": total_viewed_lessons,
            "всего просмотренное время": total_view_time,
            "процент просмотра": percentage_of_access
        })

    return jsonify(stats)
# ---------------------------------------------------------------------
