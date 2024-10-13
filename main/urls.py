from django.urls import path
from . import views



urlpatterns = [
    path('', views.home, name='home'),  # маршрут для стартовой страницы
    path('predict/', views.home, name='predict_review'),
]