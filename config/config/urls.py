"""config URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.http import HttpResponse
from django.contrib import admin
from django.urls import path
from main.views import index,analyze,chatbot_recommendation_api
def home(request):
    return HttpResponse("Django 서버가 정상 작동 중입니다 🚀")
urlpatterns = [
    path("admin/", admin.site.urls),
    path("", index),
    path("analyze/", analyze),
    path('recommend/', chatbot_recommendation_api),
]
