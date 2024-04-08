from django.urls import path
from . import views
from .views import get_csrf_token
urlpatterns=[
  path("",views.home,name="home"),
  path("index.html",views.index,name="index"),
  path("predict",views.predict,name="predict")
]