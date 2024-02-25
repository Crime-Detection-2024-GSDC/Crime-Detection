import { BACKEND_URL, requireNormalUser, getCSRF } from "./common/common_module.js";

/* 전역 변수 */
const videoTag = document.querySelector("#user-video"); // video태그
const selectionTag = document.querySelector("#camlist"); // 비디오 source 선택하는 selection태그
let err_count = 0;

/* 메인 코드 로직 */
pageMain();
async function pageMain() {
  await requireNormalUser();  // 노말유저 권한 요구
  await findAllVideoSources();
  await changeCamera();
  selectionTag.onchange = changeCamera;
  // 비디오 누르면 전체화면 되게 만들기
  videoTag.addEventListener("click", function() {videoTag.requestFullscreen();});
  // 비디오 정지 안되게 만들기
  videoTag.onpause = function () { videoTag.play(); }
}

/* 함수 */
// 모든 video input source들 조회 + select태그에 추가(오류시 false 반환)
async function findAllVideoSources() {
  let videoInputList = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind=="videoinput");
  // 없으면 에러
  if(videoInputList.length == 0) {
    alert("현재 사용가능한 카메라가 없습니다!");
    return false;
  }

  // selection태그 내에 사용 가능한 비디오 source들의 이름과 deviceId를 저장(deviceId는 dataset사용하여 저장)
    for(let i=0; i < videoInputList.length; i++) {
    let option = document.createElement("option");
    option.innerText = "Camera " + i + " " + videoInputList[i].label;
    option.dataset.deviceId = videoInputList[i].deviceId
    selectionTag.append(option);
  }
  return true;
}

// video를 select에 선택된 카메라로 변경
async function selectCamera() {
  const nowSelectedDeviceId = selectionTag.selectedOptions[0].dataset.deviceId
  let stream = null;

  stream = await navigator.mediaDevices.getUserMedia({
      video: {
          deviceId : nowSelectedDeviceId,
          width : {ideal : 320},
          height : {ideal : 240}
      }
  });
  videoTag.srcObject = stream;
	videoTag.onloadedmetadata = function() {
	  videoTag.play();
	};
}

// 현재 선택된 카메라에서 5초정도 영상을 촬영한 후, 파일을 획득하여 서버로 보냄
async function getVideo() {
  // videoMediaStream 미리 저장
  const thisMediaStream = videoTag.srcObject;
  // videoData 담아놓을 로컬 변수 리스트
  let videoData = [];

  // 1) MediaStream을 매개변수로 MediaRecorder 생성자를 호출
  let videoRecorder = new MediaRecorder(thisMediaStream, {
    mimeType: "video/webm;codec=vp8"
  });
  
  // 2) 전달받는 데이터를 처리하는 이벤트 핸들러 등록
  videoRecorder.ondataavailable = async event => {
    if(event.data?.size > 0 && err_count < 5){
      await async function() {videoData.push(event.data)}();
      videoRecorder.stop();
    }
  }

  // 3) 녹화 중지 이벤트 핸들러 등록
  videoRecorder.onstop = async () => {
    // 이 타이밍에 비디오소스가 바뀌었다면 파일 전송을 하지 않음
    if (thisMediaStream !== videoTag.srcObject) {return;}

    // 비디오 데이터 파일
    const videoBlob = new Blob(videoData, {type: "video/webm"});

    // api사용시 body는 formData로 하면 됩니다. (body = formData)
    let file = new File([videoBlob], 'video.webm', {type:'video/webm'});
    let formData = new FormData();
    formData.append("video", file);

    // 재귀실행 시작
    setTimeout(()=>{videoRecorder.start(5000);}, 0); // 비디오가 바뀌지 않은 경우에만 자기 자신을 실행

    // 백엔드 서버로 전송
    try{
        const response = await fetch(BACKEND_URL+"infer/infervideo", {
            method : "POST",
            headers : {
                'X-CSRFToken' : getCSRF(),
            },
            body : formData
        });
        await response.json();
    }
    catch(e) {
        console.log(e);
        // 백엔드서버 연결 오류 발생시 
        if((++err_count) >= 5) {
          alert("Connection to the backend server has been lost!");
            return;
        }
    }
  } 
  // 4) 녹화 시작
  videoRecorder.start(5000);
}

// video를 select에 선택된 카메라로 변경하고 녹화 루프 시작
async function changeCamera() {
  await selectCamera();
  getVideo();
}

// @ 디버그용 : blob을 다운로드
function downloadBlobAsFile(blob) {
    const recordedVideoURL = window.URL.createObjectURL(blob);
    const anchorElement = document.createElement('a');
    document.body.appendChild(anchorElement);
    anchorElement.download = 'test.webm'; // a tag에 download 속성을 줘서 클릭할 때 다운로드가 일어날 수 있도록 하기
    anchorElement.href = recordedVideoURL; // href에 url 달아주기
    anchorElement.click(); // 코드 상으로 클릭을 해줘서 다운로드를 트리거
    document.body.removeChild(anchorElement); // cleanup - 쓰임을 다한 a 태그 삭제
}