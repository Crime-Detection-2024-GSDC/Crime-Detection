"""
URL configuration for SCD_Backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.contrib import admin
from django.urls import path, include, re_path
from SCD_Backend.settings import MEDIA_URL
#from django.conf.urls.static import static
from auth_app.views import func_login, func_logout, func_change_password, func_reset_password, func_make_new_user, func_get_all_user, func_who_am_i
from infer.views import inferVideo, getVideo, getNextVideo, getPrevVideo, deleteVideo, searchPage
from rest_framework import routers
from django.views.static import serve 

router = routers.SimpleRouter()

urlpatterns = [
    path('api/auth/login', func_login),
    path('api/auth/logout', func_logout),
    path('api/auth/changepw', func_change_password),
    path('api/auth/resetpw', func_reset_password),
    path('api/auth/makeuser', func_make_new_user),
    path('api/auth/getalluser', func_get_all_user),
    path('api/auth/getme', func_who_am_i),
    path('api/infer/getvideo', getVideo),
    path('api/infer/getnextvideo', getNextVideo),
    path('api/infer/getprevvideo', getPrevVideo),
    path('api/infer/searchpage', searchPage),
    path('api/infer/deletevideo', deleteVideo),
    path('api/infer/infervideo', inferVideo),
    # https://stackoverflow.com/questions/5836674/why-does-debug-false-setting-make-my-django-static-files-access-fail
    # 정적 파일 serve는 위 주소 참고함 (원래는 nginx등 웹서버 써야하는듯)
    re_path(r'^videos/(?P<path>.*)$', serve,{'document_root': MEDIA_URL}), 
]
