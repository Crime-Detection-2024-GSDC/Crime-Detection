from rest_framework.decorators import api_view
from django.contrib.auth import authenticate, login, logout
from custom_util.custom_utils import CustomResponse, isValidPassword, isValidUsername
from django.contrib.auth.models import User
###############################
## API
import datetime

# POST : 로그인
@api_view(['POST'])
def func_login(request):
    # 이미 로그인 된 상태라면 오류
    if request.user.is_authenticated:
        return CustomResponse(
            403, 
            {"detail" : "이미 로그인 되었습니다."}
        )
    
    # 요구하는 형식(username, password) 없으면 오류
    username = request.data.get("username")
    password = request.data.get("password")
    if None in (username, password):
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )

    # 존재 안하는 유저거나 비번 틀렸으면 오류
    user = authenticate(username=username, password=password)
    if user is None: # 존재하지 않는 유저라면(혹은 비번 틀렸다면)
        return CustomResponse(
            401, 
            {"detail" : "해당 username이 존재하지 않거나, 잘못된 비밀번호입니다."}
        )

    # 이하 로그인 성공
    login(request, user)
    return CustomResponse(
        200, 
        {
            "detail" : "success",
            "username" :  "%s" % (user.username),
            "is_superuser" : user.is_superuser, # superuser인지 아닌지 확인
        }
    )

# POST : 로그아웃
@api_view(['POST'])
def func_logout(request):
    # 로그인 된 상태가 아니라면 오류
    if not request.user.is_authenticated:
        return CustomResponse(
            401, #401 Unauthorized 
            {"detail" : "로그인 된 상태가 아닙니다."}
        )

    # 로그아웃 처리
    logout(request)
    return CustomResponse(
        200, 
        {"detail" : "success"}
    )
    
# GET : 현재 로그인된 사용자의 정보 확인
@api_view(['GET'])
def func_who_am_i(request):
        # 로그인 된 상태가 아니라면 오류
    if not request.user.is_authenticated:
        return CustomResponse(
            401, #401 Unauthorized 
            {"detail" : "로그인 된 상태가 아닙니다."}
        )
    return CustomResponse(
            200, 
            {
                "detail" : "success",
                "username" : request.user.username,
                "is_superuser" : request.user.is_superuser
            }
        )

# PUT : (자기가 스스로) 패스워드 변경
@api_view(['PUT'])
def func_change_password(request):
    # 로그인 된 상태가 아니면 오류
    if not request.user.is_authenticated:
        return CustomResponse(
            401,
            {"detail" : "로그인 된 상태가 아닙니다."}
        )
    
    # 요구하는 형식(비밀번호)이 없으면 오류
    old_password = request.data.get("old_password")
    new_password = request.data.get("new_password")
    if None in (old_password, new_password):
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )

    # 새로 바꾸는 비밀번호가 옛 비밀번호랑 똑같으면 오류
    if old_password == new_password:
        return CustomResponse(
            400, 
            {"detail" : "새 비밀번호가 이전 비밀번호와 동일합니다."}
        )
    
    # 비밀번호 규칙 안맞으면 오류
    if not isValidPassword(new_password) :
        return CustomResponse(
            400, 
            {"detail" : "비밀번호 규칙이 맞지 않습니다."}
        )
    
    # 리퀘로 보낸 비밀번호가 달라도 오류
    user_test = authenticate(username=request.user.username, password=old_password)
    if user_test is None: # 존재하지 않는 유저라면(혹은 비번 틀렸다면)
        return CustomResponse(
            401, 
            {"detail" : "잘못된 비밀번호입니다."}
        )
    
    
    # 이하 패스워드 변경
    from django.contrib.auth import update_session_auth_hash
    u = User.objects.get(username=request.user.username)
    u.set_password(new_password)
    u.save()
    update_session_auth_hash(request, u)
    return CustomResponse(
        200,
        {"detail" : "success"}
    )

# POST : (어드민이 강제로) 일반 유저의 비밀번호 초기화
@api_view(['POST'])
def func_reset_password(request):
    # 로그인 된 상태가 아니면 오류
    if not request.user.is_authenticated:
        return CustomResponse(
            401,
            {"detail" : "로그인 된 상태가 아닙니다."}
        )
    
    # superuser 아니면 오류
    if not request.user.is_superuser:
        return CustomResponse(
            403,
            {"detail" : "잘못된 접근입니다."}
        )

    # 요구하는 형식(유저네임) 없으면 오류
    username = request.data.get("username")
    if None in (username, ):
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )

    # 그 user가 존재하지 않는 user라면 오류
    if len(User.objects.filter(username=username))==0:
        return CustomResponse(
            404, 
            {"detail" : "존재하지 않는 유저입니다."}
        )

    # 이하 비밀번호 변경 로직
    new_password = makeRandomPassword()
    u = User.objects.get(username=username)
    u.set_password(new_password)
    u.save()
    return CustomResponse(
        200,
        {
            "detail" : "success",
            "new_password" : new_password
        }
    )

# POST : 어드민이 일반 유저를 새로 생성
@api_view(['POST'])
def func_make_new_user(request):
    # 로그인 된 상태가 아니면 오류
    if not request.user.is_authenticated:
        return CustomResponse(
            401,
            {"detail" : "로그인 된 상태가 아닙니다."}
        )
    
    # superuser 아니면 오류
    if not request.user.is_superuser:
        return CustomResponse(
            403,
            {"detail" : "잘못된 접근입니다."}
        )

    # 요구하는 형식(유저네임) 없으면 오류
    username = request.data.get("username")
    if None in (username, ):
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )
    
    # 유저네임 형식 안맞으면 오류
    if not isValidUsername(username):
        return CustomResponse(
            400, 
            {"detail" : "username이 규칙에 맞지 않습니다."}
        )
    
    # 이미 존재하는 username이면 오류
    if len(User.objects.filter(username=username)) != 0:
        return CustomResponse(
            409, 
            {"detail" : "이미 존재하는 username입니다."}
        )

    new_password = makeRandomPassword()
    # 이하 유저 생성 로직
    new_user = User.objects.create_user(
        username, "", new_password
    )
    new_user.save()
    return CustomResponse(
        200,
        {
            "detail" : "success",
            "username" : username,
            "password" : new_password
        }
    )

# GET : 모든 유저 정보(정확히는 superuser제외) 불러오기
@api_view(['GET'])
def func_get_all_user(request):
    # 로그인 된 상태가 아니면 오류
    if not request.user.is_authenticated:
        return CustomResponse(
            401,
            {"detail" : "로그인 된 상태가 아닙니다."}
        )
    
    # superuser 아니면 오류
    if not request.user.is_superuser:
        return CustomResponse(
            403,
            {"detail" : "잘못된 접근입니다."}
        )

    # superuser를 제외한 모든 user의 리스트를 받아옴
    all_users = User.objects.filter(is_superuser=False)
    
    username_list = [user.username for user in all_users]
    userid_list = [user.id for user in all_users]

    return CustomResponse(
        200,
        {
            "detail" : "success",
            "username_list" : username_list,
            "userid_list" : userid_list
        }
    )


###
## 여기 py파일 내에서만 사용할 함수
# 첫글자는 알파벳1글자 + 나머지7글자는 숫자+특수문자+알파벳으로 된 랜덤한 비밀번호 생성
def makeRandomPassword():
    from random import choice 
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    alphabets_Plus_etc = alphabets + "1234567890~!@#$%^&*_-=+`"
    return choice(alphabets) + ''.join([choice(alphabets_Plus_etc) for _ in range(7)])