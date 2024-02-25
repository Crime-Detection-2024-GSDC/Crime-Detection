##### 라이브러리 임포트
import cv2
import numpy as np
import os
import datetime
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from SCD_Backend.settings import MEDIA_URL
from custom_util.custom_utils import CustomResponse, link
from rest_framework.decorators import api_view
from infer.models import VideoData
from django.contrib.auth.models import User
import uuid
from threading import Thread
from django.core.paginator import Paginator
import google.generativeai as genai
import PIL.Image
from io import BytesIO
from gtts import gTTS
from pygame import mixer
from time import sleep
import ffmpeg

##### 아래는 백엔드에서 코드가 아님 가능하면 수정 안하는게 좋음
# 뭔지는 모르겠는데 일단 필요 1
def getOpticalFlow(video):
    """Calculate dense optical flow of input video
    Args:
        video: the input video with shape of [frames,height,width,channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # initialize the list of optical flows
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (224, 224, 1)))

    flows = []
    for i in range(0, len(video) - 1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # Add into list
        flows.append(flow)

    # Padding the last frame as empty array
    flows.append(np.zeros((224, 224, 2)))

    return np.array(flows, dtype=np.float32)

# 뭔지는 모르겠는데 일단 필요
def farneback_visual(flows, file_path):
    # visualization farneback optical flow map
    # save the map as 'farneback_optical_flow.mp4'
    h, w = flows.shape[1:3]

    path = '/content/drive/MyDrive/solution_challange/examples'
    file_name = '/'.join(file_path.split('.')[0].split('/')[-3:]) + '_opticalFlow'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(path, file_name+'.mp4'), fourcc, 30.0, (int(w), int(h)), isColor=True)

    hsv = np.zeros((h, w, 3))
    hsv[..., 1] = 255
    for flow in flows:
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv = np.float32(hsv)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        out_video.write(np.uint8(rgb))
    out_video.release()
    return None
    pass

# 비디오 파일의 경로를 입력받은 후 Npy로 변환
def Video2Npy(file_path, resize=(224,224)):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows
    """
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    # Extract frames from video
    try:
        frames = []
        for i in range(len_frames-1):
            _, frame = cap.read()
            frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224,224,3))
            frames.append(frame)
    except:
        print("Error: ", file_path, len_frames,i)
    finally:
        frames = np.array(frames)
        cap.release()

    # Get the optical flow of video
    flows = getOpticalFlow(frames)

    # Visualize optical flow map
    optical_flow_map = farneback_visual(flows, file_path)

    result = np.zeros((len(flows),224,224,5))
    result[...,:3] = frames
    result[...,3:] = flows

    return result

# 노멀라이즈 - DataGenerator클래스에서 추출함
def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std

# 유니폼 샘플링 - DataGenerator클래스에서 추출함
def uniform_sampling(video, target_frames=64):
    # get total frames of input video and calculate sampling interval
    len_frames = int(len(video))
    interval = int(np.ceil(len_frames/target_frames))
    # init empty list for sampled video and
    sampled_video = []
    for i in range(0,len_frames,interval):
        sampled_video.append(video[i])
    # calculate numer of padded frames and fix it
    num_pad = target_frames - len(sampled_video)
    padding = []
    if num_pad>0:
        for i in range(-num_pad,0):
            try:
                padding.append(video[i])
            except:
                padding.append(video[0])
        sampled_video += padding # Add padding results
    # get sampled video
    return np.array(sampled_video, dtype=np.float32)

# 비디오 파일 로드->Npy로 변환->model함수로 추론가능하게 형태 변경
def loadVideoAndConvertToInfer(file_path):
    npyedVideo = Video2Npy(file_path)
    npyedVideo = uniform_sampling(npyedVideo, target_frames=64)
    npyedVideo[..., :3] = normalize(npyedVideo[..., :3])
    npyedVideo[..., 3:] = normalize(npyedVideo[..., 3:])
    npyedVideo = np.array(npyedVideo)
    npyedVideo = torch.tensor(npyedVideo)
    npyedVideo = npyedVideo.float()
    dataLoader = DataLoader([npyedVideo], batch_size=1, shuffle=True, num_workers=0)
    npyedVideo = next(iter(dataLoader)) 
    return npyedVideo

# 모델 자체에 대한 정의
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.relu=nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool3d(kernel_size=(8, 1 ,1), stride=(8, 1, 1))

        # Construct block of RGB layers which takes RGB channel(3) as input
        self.rgb_conv1 = nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.rgb_conv2 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.rgb_maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.rgb_conv3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.rgb_conv4 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.rgb_maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.rgb_conv5 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.rgb_conv6 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.rgb_maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.rgb_conv7 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.rgb_conv8 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.rgb_maxpool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Construct block of optical flow layers which takes the optical flow channel(2) as input
        self.opt_conv1 = nn.Conv3d(2, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.opt_conv2 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.opt_maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.opt_conv3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.opt_conv4 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.opt_maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.opt_conv5 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.opt_conv6 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.opt_maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.opt_conv7 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.opt_conv8 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.opt_maxpool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Construct merging Block
        self.merge_conv1 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.merge_conv2 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.merge_maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.merge_conv3 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.merge_conv4 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.merge_maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.merge_conv5 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.merge_conv6 = nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.merge_maxpool3 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3))

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

        # Initialize weights
        self.__init_weight()

    def forward(self, x):
        x = x.transpose(2, 4)
        x = x.transpose(3, 4)
        x = x.transpose(1, 2)
        rgb = x[:,:3,:,:,:]
        opt = x[:,3:5,:,:,:]

        # Pass through the RGB data through the blocks of RGB layers
        rgb = self.rgb_conv1(rgb)
        rgb = self.relu(rgb)
        rgb = self.rgb_conv2(rgb)
        rgb = self.relu(rgb)
        rgb = self.rgb_maxpool1(rgb)
        rgb = self.rgb_conv3(rgb)
        rgb = self.relu(rgb)
        rgb = self.rgb_conv4(rgb)
        rgb = self.relu(rgb)
        rgb = self.rgb_maxpool2(rgb)

        rgb = self.rgb_conv5(rgb)
        rgb = self.relu(rgb)
        rgb = self.rgb_conv6(rgb)
        rgb = self.relu(rgb)
        rgb = self.rgb_maxpool3(rgb)
        rgb = self.rgb_conv7(rgb)
        rgb = self.relu(rgb)
        rgb = self.rgb_conv8(rgb)
        rgb = self.relu(rgb)
        rgb = self.rgb_maxpool4(rgb)

        # Pass through the optical flow data through the blocks of RGB layers
        opt = self.opt_conv1(opt)
        opt = self.relu(opt)
        opt = self.opt_conv2(opt)
        opt = self.relu(opt)
        opt = self.opt_maxpool1(opt)
        opt = self.opt_conv3(opt)
        opt = self.relu(opt)
        opt = self.opt_conv4(opt)
        opt = self.relu(opt)
        opt = self.opt_maxpool2(opt)

        opt = self.opt_conv5(opt)
        opt = self.relu(opt)
        opt = self.opt_conv6(opt)
        opt = self.relu(opt)
        opt = self.opt_maxpool3(opt)
        opt = self.opt_conv7(opt)
        opt = self.sigmoid(opt)
        opt = self.opt_conv8(opt)
        opt = self.sigmoid(opt)
        opt = self.opt_maxpool4(opt)

        # Fuse by performing elementwise multiplication of rgb and opt tensors.
        fused = rgb * opt
        # Perform maxpooling of fused
        fused = self.maxpool(fused)

        # Pass through the fused data into merging block
        merged = self.merge_conv1(fused)
        merged = self.relu(merged)
        merged = self.merge_conv2(merged)
        merged = self.relu(merged)
        merged = self.merge_maxpool1(merged)
        merged = self.merge_conv3(merged)
        merged = self.relu(merged)
        merged = self.merge_conv4(merged)
        merged = self.relu(merged)
        merged = self.merge_maxpool2(merged)
        merged = self.merge_conv5(merged)
        merged = self.relu(merged)
        merged = self.merge_conv6(merged)
        merged = self.relu(merged)
        merged = self.merge_maxpool3(merged)

        # Fully Connected Layers
        x = merged.view(merged.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Perform weight initialization ("kaiming normal")
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    #nn.init.constant_(m.bias, 0)
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / torch.sqrt(torch.tensor(fan_in))
                    nn.init.uniform_(m.bias, -bound, bound)

##### 상수 영역
# 모델 주소
MODEL_PATH = "custom_util/ViolenceModel.pth"

# GEMINI 사용 여부
USE_GEMINI = True

# 구글 GEMINI의 API키가 적힌 파일의 경로 및 이름
API_KEY_FILE_PATH :str = "GOOGLE_GEMINI_API_KEY.txt"

# 모델 설정들
GENERATION_CONFIG = {
    "temperature" : 0,
    "top_p" : 1,
    "top_k" : 1,
    "max_output_tokens" : 400,
}
SAFETY_SETTINGS = [{"category" : i,"threshold" : "BLOCK_NONE"} for i in [("HARM_CATEGORY_%s" % cat) for cat in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]]]

# 프롬프트에 넣을 질문
PROMPT_MESSAGE :str = "Based on the frames of this video, what is currently happening? Please summarize them briefly."

# (디버깅용) 추론 API 디버그 여부 (True면 영상이 항상 폭력적이라고 판단하게 됨)
DEBUG_INFER = False

##### 전역 변수
# 모델 로드
print("## Model loading")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# gemini 사용 여부
useGemini = USE_GEMINI

# genai configure
print("## Configuring Gemini")
# API키 입력
if useGemini:
    try:
        with open(API_KEY_FILE_PATH, 'r') as f:
            API_KEY :str = f.readline()
            genai.configure(api_key=API_KEY)
    except:
        print("Error: An error occurred in the Gemini settings. Therefore, Gemini is not being used.")
        useGemini = False

##### REST API 정의
# 동영상 하나를 전달받아서 저장 (모델에 넣고 검출때리는건 속도 문제로 인해 타 스레드에서 진행)
@api_view(['POST'])
def inferVideo(request):
    # 로그인 안 되어 있다면 에러
    if not request.user.is_authenticated:
        return CustomResponse(401, {'detail' : "로그인되어있지 않음" })

    # post로 보낸 파일은 request.FILES안에 들어감
    videoBlob = request.FILES.get('video')
    # 'video'라는 이름으로 파일 업로드 안했을 경우에는 에러
    if videoBlob == None: return CustomResponse(400, {'detail' : "파일이 없음" })

    # 일단 VideoData 하나 생성
    videoData:VideoData = VideoData()
    
    # 이거 찍은 user가 누구인지 넣어줌
    videoData.user = request.user
    # 폭력성 여부는 일단 False를 기본값으로
    videoData.isViolent = False
    # 생성할 파일 이름은 uuid로 랜덤하게 생성
    uuidName:str = str(uuid.uuid4())
    videoData.filename = uuidName+'.mp4'
    # pk를 얻기 위해 일단 저장함
    videoData.save()
    pk:int = videoData.id
    # auto_now_add=True라서 picture_data생성시에 안에 이미 날짜가 있음
    d = videoData.date
    # 현재 날짜(20230101형식)를 폴더 이름으로 함
    folder_name:str = "%04d%02d%02d" % (d.year, d.month, d.day)
    # 파일경로
    path_without_file:str = MEDIA_URL.rstrip('/') + "/" + folder_name
    # 파일경로 + 파일 이름(mp4)
    filePath:str = path_without_file + "/" + videoData.filename
    # 해당 파일경로가 존재하지 않는 경로면 만듦
    if not os.path.exists(path_without_file):
        os.makedirs(path_without_file)
    
    # POST로 받은 파일 저장함
    webmFilePointer = BytesIO() # webm 파일 넣어놓을 파일 포인터
    for chunk in videoBlob.chunks():
        webmFilePointer.write(chunk)
    webmFilePointer.seek(0)
    inputData = webmFilePointer.read()
    process = (ffmpeg.input("pipe:").output(filePath).overwrite_output().run_async(pipe_stdin=True, quiet=True))
    process.communicate(input=inputData)
    process.wait()
    webmFilePointer.close()

    # 모델 추론을 위해 쓰레드 생성 후 시작
    Thread(target = threadWork, args=(pk, filePath)).start()
    # 응답
    res = { "detail" : "success" }
    # 끝
    return CustomResponse(200, res)

# 영상의 id를 받아서 해당 영상의 검출결과 및 영상 경로 확인
@api_view(['POST'])
def getVideo(request):
    # 로그인 안 되어 있다면 에러
    if not request.user.is_authenticated:
        return CustomResponse(401, {'detail' : "로그인되어있지 않음" })
    # superuser 아니면 오류
    if not request.user.is_superuser:
        return CustomResponse(403, {"detail" : "잘못된 접근입니다."})

    # id 없으면 오류
    id = request.data.get("id")
    if None in (id, ):
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )

    videoData:VideoData = None
    try:
        videoData = VideoData.objects.get(id=id)
    except:
        return CustomResponse(404, {"detail" : "존재하지 않는 영상"})
    
    # 유저 이름 검출    
    username:str = videoData.user.username
    # 시간 및 폴더 이름 검출
    date = videoData.date
    folderName:str = "%04d%02d%02d" % (date.year, date.month, date.day)
    str_dateTime:str = "%04d-%02d-%02d %02d:%02d:%02d" % (date.year, date.month, date.day, date.hour, date.minute, date.second)

    # 파일 이름 검출(확장자 붙음)
    filename = videoData.filename

    # 응답 반환
    return CustomResponse(
        200,
        {
            "detail" : "success",
            "username" : username,
            "date" : str_dateTime,
            "nowPath" : "/videos/"+folderName+"/"+filename,
            "isViolent" : bool(videoData.isViolent)
        }
    )

# 영상의 id, username를 받아서 
# 해당 영상의 다음 순서에 있는 영상(해당 사진을 찍은 user가 찍은 다음 영상)의 
# 검출결과 및 정보, 다음 영상의 id 확인
@api_view(['POST'])
def getNextVideo(request):
    # 로그인 안 되어 있다면 에러
    if not request.user.is_authenticated:
        return CustomResponse(401, {'detail' : "로그인되어있지 않음" })
    # superuser 아니면 오류
    if not request.user.is_superuser:
        return CustomResponse(403, {"detail" : "잘못된 접근입니다."})

    # id 없으면 오류
    id = request.data.get("id")
    if None in (id, ):
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )

    videoData:VideoData = None
    try:
        videoData = VideoData.objects.get(id=id)
    except:
        return CustomResponse(404, {"detail" : "존재하지 않는 영상"})

    # 다음 사진 확인 (없으면 에러)
    nextVideoData = VideoData.objects.filter(user=videoData.user, id__gt=videoData.id).order_by('id').first()
    if nextVideoData == None:
        return CustomResponse(404, {"detail" : "다음 영상이 없음"})
    # 다음 영상 id 검출
    nextId:int = nextVideoData.id

    # 유저 이름 검출    
    username:str = nextVideoData.user.username
    # 시간 및 폴더 이름 검출
    date = nextVideoData.date
    folderName:str = "%04d%02d%02d" % (date.year, date.month, date.day)
    str_dateTime:str = "%04d-%02d-%02d %02d:%02d:%02d" % (date.year, date.month, date.day, date.hour, date.minute, date.second)

    # 파일 이름 검출(확장자 붙음)
    filename = nextVideoData.filename

    # 응답 반환
    return CustomResponse(
        200,
        {
            "detail" : "success",
            "nextId" : nextId,
            "username" : username,
            "date" : str_dateTime,
            "nextPath" : "/videos/"+folderName+"/"+filename,
            "isViolent" : bool(nextVideoData.isViolent)
        }
    )

# 영상의 id, username를 받아서 
# 해당 영상의 이전 순서에 있는 영상(해당 사진을 찍은 user가 찍은 이전 영상)의 
# 검출결과 및 정보, 이전 영상의 id 확인
@api_view(['POST'])
def getPrevVideo(request):
    # 로그인 안 되어 있다면 에러
    if not request.user.is_authenticated:
        return CustomResponse(401, {'detail' : "로그인되어있지 않음" })
    # superuser 아니면 오류
    if not request.user.is_superuser:
        return CustomResponse(403, {"detail" : "잘못된 접근입니다."})

    # id 없으면 오류
    id = request.data.get("id")
    if None in (id, ):
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )

    videoData:VideoData = None
    try:
        videoData = VideoData.objects.get(id=id)
    except:
        return CustomResponse(404, {"detail" : "존재하지 않는 영상"})

    # 다음 사진 확인 (없으면 에러)
    prevVideoData = VideoData.objects.filter(user=videoData.user, id__lt=videoData.id).order_by('-id').first()
    if prevVideoData == None:
        return CustomResponse(404, {"detail" : "이전 영상이 없음"})
    # 다음 영상 id 검출
    prevId:int = prevVideoData.id

    # 유저 이름 검출    
    username:str = prevVideoData.user.username
    # 시간 및 폴더 이름 검출
    date = prevVideoData.date
    folderName:str = "%04d%02d%02d" % (date.year, date.month, date.day)
    str_dateTime:str = "%04d-%02d-%02d %02d:%02d:%02d" % (date.year, date.month, date.day, date.hour, date.minute, date.second)

    # 파일 이름 검출(확장자 붙음)
    filename = prevVideoData.filename

    # 응답 반환
    return CustomResponse(
        200,
        {
            "detail" : "success",
            "prevId" : prevId,
            "username" : username,
            "date" : str_dateTime,
            "prevPath" : "/videos/"+folderName+"/"+filename,
            "isViolent" : bool(prevVideoData.isViolent)
        }
    )

# 영상 id 받아서 삭제
@api_view(['DELETE'])
def deleteVideo(request):
    # 로그인 안 되어 있다면 에러
    if not request.user.is_authenticated:
        return CustomResponse(401, {'detail' : "로그인되어있지 않음" })
    # superuser 아니면 오류
    if not request.user.is_superuser:
        return CustomResponse(403, {"detail" : "잘못된 접근입니다."})
    # 영상 id 없으면 오류
    id = request.data.get("id")
    if None in (id, ):
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )
    # id에 해당하는 영상이 없다면 에러
    videoData:VideoData = None
    try:
        videoData = VideoData.objects.get(id=id)
    except:
        return CustomResponse(
            404, 
            {"detail" : "존재하지 않는 영상"}
        )
    
    # 이전, 이후 사진의 id를 반환하기 위한 부분
    prevVideo = VideoData.objects.filter(user=videoData.user, id__lt=id).order_by('-id').first()
    nextVideo = VideoData.objects.filter(user=videoData.user, id__gt=id).order_by('id').first()
    nextVideoId = VideoData.id if nextVideo != None else None
    prevVideoId = VideoData.id if prevVideo != None else None

    # 폴더 이름 검출
    date = videoData.date
    folderName:str = "%04d%02d%02d" % (date.year, date.month, date.day)
    # 파일 이름
    fileName:str = videoData.filename
    # 폴더 경로
    folderPath:str = MEDIA_URL.rstrip("/")+"/"+folderName
    # 파일 경로
    filePath:str = folderPath+"/"+fileName
    # 실물 파일 삭제
    if os.path.isfile(filePath):
        os.remove(filePath)
    else:
        print(f"경고 : 존재하지 않는 파일({filePath})")
    if os.path.exists(folderPath):
        # 파일 삭제 후 더 이상 해당 날짜 폴더에 사진이 없다면 해당 폴더도 삭제
        if not list(filter(lambda x : (x.endswith(".mp4")), os.listdir(folderPath))):
            os.rmdir(folderPath)
    else:
        print(f"경고 : 존재하지 않는 폴더({folderPath})")

    # DB에서 객체 삭제
    videoData.delete()
    # 결과 반환
    return CustomResponse(200, {
        "detail" : "success",
        "nextVideoId" : nextVideoId,
        "prevVideoId" : prevVideoId
    })

# 영상 검색
@api_view(['POST'])
def searchPage(request):
    # 로그인 안 되어 있다면 에러
    if not request.user.is_authenticated:
        return CustomResponse(401, {'detail' : "로그인되어있지 않음" })
    # superuser 아니면 오류
    if not request.user.is_superuser:
        return CustomResponse(403, {"detail" : "잘못된 접근입니다."})

    # usernames (유저명(문자열)의 리스트)
    # date ('2023-11-15'형식 문자열)
    # timeStart ('00:10:30' 형식 문자열)
    # timeEnd ('21:36:30' 형식 문자열)
    # viewCounts (숫자. 20, 40, 60, 80, 100)
    # index (숫자)
    # searchAll (불리언값)
    # 위 내용들 모두 없다면 에러
    usernames, date, timeStart, timeEnd, viewCounts, searchAll = (
        None, None, None, None, None, None
    )

    try:
        usernames = request.data.get("usernames")
        date = datetime.datetime.strptime(request.data.get("date"), '%Y-%m-%d')
        timeStart = list(map(int, request.data.get("timeStart").split(":")))
        timeEnd = list(map(int, request.data.get("timeEnd").split(":")))
        viewCounts = request.data.get("viewCounts")
        index = request.data.get("index")
        searchAll = request.data.get("searchAll")
        if None in (usernames, date, timeStart, timeEnd, viewCounts, index, searchAll):
            raise ValueError
        if viewCounts not in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            raise ValueError
    except:
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다."}
        )

    dateStart = date.replace(hour=timeStart[0], minute=timeStart[1], second=timeStart[2])
    dateEnd = date.replace(hour=timeEnd[0], minute=timeEnd[1], second=timeEnd[2]) + datetime.timedelta(seconds=1)

    user_lists = [User.objects.get(username=name) for name in usernames]    
    # userIDs에 있는 유저들로부터 온 영상이면서
    # dateStart와 dateEnd사이에 있으면서
    # searchAll이 True면 isViolence값이 0이상(모든)인 객체 전부 들고오기.
    # searchAll이 False면 isViolence값이 1이상인 객체만 들고오기.
    minViolenceValue:int = 0 if searchAll==True else 1
    search_lists = VideoData.objects.filter(
        user__in=user_lists,
        date__gte=dateStart,
        date__lt=dateEnd,
        isViolent__gte=minViolenceValue
    )

    # 반환 결과 준비
    responseData = { "detail" : "success" , "results" : []}

    # 페이지네이션 준비
    paginator = Paginator(search_lists, viewCounts)

    # 총 페이지 확인
    responseData['totalIndex'] = paginator.num_pages

    # 정상적인 페이지범위가 아니여도 에러 
    if not (1 <= index <= responseData['totalIndex']) :
        return CustomResponse(
            400, 
            {"detail" : "요청 형식이 잘못되었습니다. (잘못된 페이지 범위)"}
        )
    
    # 쿼리셋 구하기
    queryset = paginator.get_page(index).object_list

    # 반환 결과 result리스트에 형식 맞추어 추가
    for v in queryset:
        video:VideoData = v
        #사진id, 유저명, 시간, 폭력성 여부 형식의 리스트의 리스트가 results임
        tmp = {}
        tmp['id'] = video.id
        tmp['username'] = video.user.username
        tmp['date'] = video.date.strftime("%Y-%m-%d %H:%M:%S")
        tmp['isViolent'] = video.isViolent
        responseData['results'].append(tmp)
    
    return CustomResponse(200, responseData)

##### 기타 함수 정의
# 영상의 pk와 path를 받아서 모델 검출 후 DB 수정 + 결과 출력
def inferVideoByModel(pk:int, filePath:str):
    # pk에 맞는 VideoData객체 DB에서 꺼내기 시도
    videoData:VideoData = None
    try:
        videoData = VideoData.objects.get(id=pk)
    except: # DB에 없다면 에러
        print(f"오류 : DB에 영상이 없음. id : {pk} / filePath : {filePath}")
        return
    # 저장한 비디오를 Npy파일로 불러오기 
    npyedVideo = loadVideoAndConvertToInfer(filePath)
    npyedVideo = npyedVideo.to(device)
    # 모델에 넣어서 추론
    output = model(npyedVideo)
    _, predicted = torch.max(output, 1)
    isViolent:bool = (predicted[0] == 1) or DEBUG_INFER
    #if DEBUG_INFER:
    #    print("@@ 디버그: DEBUG_INFER값이 True이므로, 이 영상은 모델 판단 여부와 상관없이 반드시 폭력적이라고 판단합니다.")
    # 모델의 isViolent정보 수정
    videoData.isViolent = isViolent
    videoData.save()

    date = videoData.date
    folderName:str = "%04d%02d%02d" % (date.year, date.month, date.day)
    serveFilePath:str = "http://localhost:8000/videos/"+folderName+'/'+videoData.filename

    #print(f"@debug/isViolent : {isViolent}")
    # 영상이 폭력적이라고 판단되었을 때 실행할 뭐시기
    if isViolent:
        user:User = videoData.user
        print(f"Video sent by user <{user.username}> is judged to be violent.")
        print("Link (Ctrl+left click): " + link(serveFilePath))
        return (True, videoData.user, serveFilePath)
    
    return (False, None, serveFilePath)

# 영상의 path를 받아서 구글 Gemini로 보내어 문맥 분석 시킴
def analyzeVideoByGemini(filePath:str):
    # 제미니 모델 지정
    model = genai.GenerativeModel(
        model_name='gemini-pro-vision',
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )

    # 비디오 객체 저장할 리스트
    videoObjs = []

    # 파일 포인터(나중에 해제해야함) 리스트
    pseudoPointers = []

    # 비디오 파일 오픈
    video = cv2.VideoCapture(filePath)
    while(video.isOpened()):
        ret, image = video.read()
        if image is None : break
        if (int(video.get(1)) % 16 == 0):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
            b = BytesIO()
            image.save(b,format="jpeg")
            jpegImage = PIL.Image.open(b)
            videoObjs.append(jpegImage)
            pseudoPointers.append(b)
    video.release()

    # 프롬프트 준비
    encodedPrompt = [PROMPT_MESSAGE, *(videoObjs[:10])]
    
    # Gemini 응답 출력
    response = model.generate_content(encodedPrompt,stream=False)
    response.resolve()
    # 메모리 내 포인터 해제
    for p in pseudoPointers:
        p.close()

    # 프롬프트 결과 반환
    return response.text

# 문자열을 gtts를 사용하여 사운드로 출력
def playTTS(prompt :str) :
    soundObj = gTTS(prompt, lang='en')
    # 소리 출력용 가짜 파일 포인터
    b = BytesIO()
    soundObj.write_to_fp(b) #mp3파일 형식임    
    # https://stackoverflow.com/questions/51164040/gtts-direct-output 참고
    mixer.init()
    b.seek(0)
    mixer.music.load(b, "mp3")
    mixer.music.play()
    while mixer.music.get_busy():
        sleep(1) 
    # 소리 출력용 가짜 파일 포인터 해제
    b.close()

# (쓰레드 함수 자체) 저장한 영상의 pk와 path를 받아서 모델 검출 후 DB 수정 
def threadWork(pk:int, filePath:str):
    result, userName, serveFilePath = inferVideoByModel(pk, filePath)
    
    if not (result and useGemini): return

    promptResponse :str = None
    try:
        promptResponse :str = analyzeVideoByGemini(filePath)
        print(f"AI analysis of a video that <{userName}> has sent judged to be violent. Here is the analysis summary:\n{promptResponse}\nLink (Ctrl+left click): {link(serveFilePath)}")
        playTTS(promptResponse)
    except:
        print("## Error in the Gemini usage logic!!")
        return