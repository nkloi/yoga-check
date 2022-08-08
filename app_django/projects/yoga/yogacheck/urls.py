from django.urls import path  
from . import views
  
app_name = 'yogacheck'  
urlpatterns = [  
    path('image_request', views.image_request, name = "image_request") 
]