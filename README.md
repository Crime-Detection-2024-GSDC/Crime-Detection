# 👬 Crime Detection (crime data extraction and modeling)

## Introduction

### 🌟 Team - Crime Detection

#### 🔅 Members

|                                        Dong Ha KANG                                         |                                         Chan Won KIM                                         |                                          Yun Ho BAE                                          |                                         Tak Hyun LEE                                         |
| :-----------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| <img src='https://avatars.githubusercontent.com/u/57825834?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/123648087?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/126548916?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/144776756?v=4' height=80 width=80px></img> |
|                         [Github](https://github.com/EasternPen9uin)                         |                            [Github](https://github.com/chanwon0)                             |                            [Github](https://github.com/uyunho99)                             |                            [Github](https://github.com/Kongtaks)                             |

#### 🔅 Contribution

- [`Dong Ha KANG`] &nbsp; Django, Back API Server, DB test
- [`Chan Won KIM`] &nbsp; Model Research, ML/DL
- [`Yun Ho BAE`]&nbsp; Model Research, ML/DL
- [`Tak Hyun LEE`] &nbsp; JS, Front Web, UI

## Project Outline

### 🎯 Project Goal

- GOAL
  - Implementing features for hazardous area data collection
  - Modeling learning through features
- Function
  - Shooting a short video with a camera
  - Identify and learn risky behaviors from short videos
  - Video management possible in administrator mode


## Server Requirements
1. It is recommended to operate on a PC with a graphics card that supports CUDA operations. (Operation is possible with CPU, but speed issues may arise.)
2. The code logic involves video encoding using ffmpeg (using the ffmpeg-python library). Therefore, please install ffmpeg in advance.
```Link: https://ffmpeg.org/download.html```
3. The code logic requires an API key from Google Gemini. Please follow the instructions below after obtaining the API key from the link provided. (Take care not to leak the API key.)
```Link: https://aistudio.google.com/app/prompts/new_chat?hl=ko```
4. Python version 3.9 or higher is required.
5. ANSI escape codes are used. However, the Windows Command Prompt (cmd.exe) does not support ANSI escape codes properly without registry modifications. If using Windows, please consider using PowerShell or modify the relevant registry values (VirtualTerminalLevel).
6. This frontend server uses nginx. Please install nginx.
```Link: https://www.nginx.com/resources/wiki/start/topics/tutorials/install/```


## 🔨 How to Use
### Project Download
```git clone https://github.com/Crime-Detection-2024-GSDC/Crime-Detection```

### Initial Setup (Backend)
1. Use the following command to install Python libraries:
```pip install -r requirements.txt```
2. Install ffmpeg.
3. Input the API key obtained from Google Gemini into the file GOOGLE_GEMINI_API_KEY.txt. Be cautious not to expose the API key.

### Initial Setup (Frontend) 
An SSL certificate is required for HTTPS connection (as the navigator.mediaDevices function demands HTTPS). However, the actual process of obtaining a certificate can be complex, so it is not detailed here. Below is a method of issuing a self-signed certificate using OpenSSL.
1. Install OpenSSL.
```Link: https://www.openssl.org/source/```
2. After installation, enter the following commands in sequence:
```openssl genrsa -out privKey.pem 1024```  
```openssl req -new -key privKey.pem -out private-csr.pem```  
```openssl x509 -req -days 730 -in private-csr.pem -signKey privKey.pem -out cert.pem```  
3. Place the resulting cert.pem and privKey.pem files into the "cert" folder inside the "Frontend" folder.
4. After completing the above steps, overwrite the contents of the "Frontend" folder in the directory where nginx is installed.
 
### Run (Backend)
1. Go to Backend folder
2. Enter the following command:
```daphne -b 0.0.0.0 -p 8000 SCD_Backend.asgi:application```

### Run (Frontend)
1. Go to nginx folder
2. Enter the following command:
```nginx```

### Access in your browser:
Enter "https://(server IP)" in the browser address bar.

## Miscellaneous
1. Videos are saved in the Backend/savedVideos folder.
2. The initial password for the administrator account (username: admin) is "admin"