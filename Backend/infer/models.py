from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class VideoData(models.Model):
    # PK
    id = models.BigAutoField(primary_key=True)
    # 사진 누가 올렸나
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    # 올린 날짜
    date = models.DateTimeField(auto_now_add=True)
    # 파일 이름
    filename = models.CharField(max_length=150)
    # 폭력성 여부
    isViolent = models.IntegerField()