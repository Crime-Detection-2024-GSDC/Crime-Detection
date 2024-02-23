from rest_framework.response import Response
import os

def CustomResponse(status_code : int, response_without_code):
    response_without_code['status_code'] = status_code
    return Response(response_without_code, status=status_code)

def isValidUsername(arg_username):
    import re
    username_regex = re.compile(r'^[a-zA-Z0-9_]{3,20}$')
    return bool(username_regex.match(arg_username))

def isValidPassword(arg_password):
    import re
    password_regex = re.compile(r'^(?=.*[a-zA-Z])(?=.*[!@#$%^*+=_-])(?=.*[0-9])[a-zA-Z0-9!@#$%^*+=_-]{7,30}$')
    return bool(password_regex.match(arg_password))

# Ctrl+마우스를 눌러서 바로 열 수 있는 링크 생성
def link(uri, label=None):
    if label is None: 
        label = uri
    parameters = ''
    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)