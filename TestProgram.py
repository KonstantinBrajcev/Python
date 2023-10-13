from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=255)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    users = models.ManyToManyField(User, related_name='products')
    lessons = models.ManyToManyField('Lesson', related_name='products')

class Lesson(models.Model):
    name = models.CharField(max_length=255)
    video_link = models.URLField()
    duration = models.IntegerField()
    products = models.ManyToManyField(Product, related_name='lessons')

class UserLesson(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE)
    view_time = models.IntegerField(default=0)
    status = models.CharField(max_length=20, default='Не просмотрено')

class User(models.Model):
    name = models.CharField(max_length=255)

# Далее нужно создать представления и URL маршруты для обработки запросов API
# Например, для получения списка уроков для конкретного продукта:

from django.shortcuts import get_object_or_404
from rest_framework import generics
from .models import Product, Lesson, UserLesson
from .serializers import LessonSerializer

class LessonList(generics.ListAPIView):
    serializer_class = LessonSerializer

    def get_queryset(self):
        product_id = self.kwargs['product_id']
        product = get_object_or_404(Product, id=product_id)
        return product.lessons.all()

# В файле urls.py нужно добавить маршрут для этого представления:
from django.urls import path
from .views import LessonList

urlpatterns = [
    path('products/<int:product_id>/lessons/', LessonList.as_view(), name='lesson-list')]

# Для получения списка уроков для конкретного пользователя и продукта:

class UserLessonList(generics.ListAPIView):
    serializer_class = UserLessonSerializer

    def get_queryset(self):
        username = self.kwargs['username']
        product_name = self.kwargs['product_name']
        user = get_object_or_404(User, name=username)
        product = get_object_or_404(Product, name=product_name)
        user_lessons = UserLesson.objects.filter(user=user, lesson__products=product)
        return user_lessons

# В файле urls.py нужно добавить маршрут для этого представления:
from django.urls import path
from .views import UserLessonList

urlpatterns = [
    path('users/<str:username>/products/<str:product_name>/lessons/', UserLessonList.as_view(), name='user-lesson-list'),
]

# Для получения статистики по продуктам:

from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def get_stats(request):
    products = Product.objects.all()
    stats = []
    
    for product in products:
        users_count = product.users.count()
        lessons_count = product.lessons.count()
        total_viewed_lessons = 0
        total_view_time = 0
        
        for user_lesson in UserLesson.objects.filter(lesson__products=product):
            if user_lesson.status == 'Просмотрено':
                total_viewed_lessons += 1
                total_view_time += user_lesson.view_time
        
        percentage_of_access = (users_count / User.objects.count()) * 100
        
        stats.append({
            'имя продукта': product.name,
            'счетчик пользователей': users_count,
            'Счетчик уроков': lessons_count,
            'всего просмотрено уроков': total_viewed_lessons,
            'всего просмотренное время': total_view_time,
            'процент просмотра': percentage_of_access,
        })
    
    return Response(stats)

# В файле urls.py нужно добавить маршрут для этого представления:
from django.urls import path
from .views import get_stats

urlpatterns = [
    path('stats/', get_stats, name='stats'),
]