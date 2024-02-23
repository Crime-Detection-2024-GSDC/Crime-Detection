import { drawBoxInCanvas, sendMyRequest } from "./common/common_module.js";

const videoField = document.querySelector("#viewVideo"); 
const dateTime = document.querySelector("#dateTime");
const username = document.querySelector("#username");
const isViolentField = document.querySelector("#isViolentField");
const prev = document.querySelector("#prev");
const next = document.querySelector("#next");
const pkText = document.querySelector("#pkText");
const deleteBtn = document.querySelector("#deleteBtn");

let videoPK = null;
let isViolent = null;
let nowPath = null;
let username_text = null;
let date_text = null;

pageMain();
async function pageMain() {
    videoPK = localStorage.getItem("videoPK");
    // Search사이트에서 넘어온게 아니라면
    if(videoPK===null) {
        alert("올바르지 못한 접근입니다.");
        return;
    }
    await showVideo(videoPK);
    prev.onclick = async () => {await prevBtn();};
    next.onclick = async () => {await nextBtn();};
    deleteBtn.onclick = async () => {await deleteImage();};
    localStorage.removeItem("videoPK");
    
}

async function showVideo(arg_id) {
    videoPK = arg_id;
    const body = { id : videoPK };
    const responseJson = await sendMyRequest("infer/getvideo", "POST", body);
    if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패
		alert(responseJson.detail);
		return;
	}
    username_text = responseJson.username;
    date_text = responseJson.date;
    isViolent = responseJson.isViolent;
    nowPath = responseJson.nowPath;
    videoField.src = nowPath
    changeText();
    undisableAllBtns();
}

async function prevBtn() {
    disableAllBtns();
    const body = { id : videoPK}
    const responseJson = await sendMyRequest("infer/getprevvideo", "POST", body);
    if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패
		alert(responseJson.detail);
        undisableAllBtns();
		return;
	}

    videoPK = responseJson.prevId;
    username_text = responseJson.username;
    date_text = responseJson.date;
    nowPath = responseJson.prevPath;
    videoField.src = nowPath
    isViolent = responseJson.isViolent;
    changeText();
    undisableAllBtns();
}

async function nextBtn() {
    disableAllBtns();
    const body = { id : videoPK}
    const responseJson = await sendMyRequest("infer/getnextvideo", "POST", body);
    if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패
		alert(responseJson.detail);
		undisableAllBtns();
        return;
	}

    videoPK = responseJson.nextId;
    username_text = responseJson.username;
    date_text = responseJson.date;
    nowPath = responseJson.nextPath;
    videoField.src = nowPath
    isViolent = responseJson.isViolent;
    changeText();
    undisableAllBtns();
}

async function deleteImage() {
    // 생성할건지 묻는 창 띄우가
	if(confirm(`영상을 정말로 삭제하시겠습니까?`) == false) {
		return;	
	}
    const body = { id : videoPK};
    const responseJson = await sendMyRequest("infer/deletevideo", "DELETE", body);
    if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패
		alert(responseJson.detail);
		return;
	}
    alert("영상이 삭제되었습니다!");
    if(responseJson.nextVideoId != null) {
        showPic(responseJson.nextVideoId);
    }
    else if(responseJson.prevVideoId != null) {
        showPic(responseJson.prevVideoId);
    }
    else {
        alert("더 이상 남아있는 영상이 없습니다!");
    }
}

function changeText() {
    dateTime.textContent = date_text;
    username.textContent = username_text;
    isViolentField.textContent = isViolent ? "O" : "X";
    pkText.textContent = videoPK;
}

function disableAllBtns() {
    next.disabled = true;
    prev.disabled = true;
}

function undisableAllBtns() {
    next.disabled = false;
    prev.disabled = false;
}