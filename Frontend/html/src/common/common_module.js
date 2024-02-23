////// 각 js파일에서 공통적으로 사용할 변수, 함수들
export const BACKEND_URL = "https://" + location.host + "/api/";
export const BACKEND_WS_URL = "wss://" + location.host + "/ws-api/";

// csrftoken 쿠키 검출용 함수
export function getCookie(cookieName) {
	cookieName = `${cookieName}=`;
	let cookieData = document.cookie;
	let cookieValue = "";
	let start = cookieData.indexOf(cookieName);
	if (start !== -1) {
	  start += cookieName.length;
	  let end = cookieData.indexOf(";", start);
	  if (end === -1) end = cookieData.length;
	  cookieValue = cookieData.substring(start, end);
	}
	return cookieValue;
}

export function getCSRF() {
	return getCookie("csrftoken");
}

// 백엔드 서버로 csrf토큰 넣어서 요청 보내기
export async function sendMyRequest(path, method, body_data) {
    const url = BACKEND_URL;
	const response = await fetch(url + path, {
		method: method,
		headers: {
			'Content-Type': 'application/json',
			'X-CSRFToken' : getCSRF() 
		},
		body: JSON.stringify(body_data),
		credentials : 'include',
	});
	const responseJson = await response.json();
    return responseJson;
}

export async function sendMyFile(path, method, body_data) {
    const url = "https://" + location.host + "/api/";
	const response = await fetch(url + path, {
		method: method,
		headers: {
			'Content-Type': 'multipart/form-data',
			'X-CSRFToken' : getCSRF()
		},
		body: JSON.stringify(body_data),
		credentials : 'include',
	})
	
	const responseJson = await response.json();
    return responseJson;
}


// 로그아웃 요청 보내기 + 로그인 페이지로 보내버리기
export async function sendLogoutRequest() {
	const responseJson = await sendMyRequest("auth/logout", "POST");
    if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패시
		alert(responseJson.detail)
        return;
    }
	alert(`로그아웃 되었습니다.`);
    window.location.href = "login.html";
}

// 유저 객체가 포함된 요청 보내기
export async function sendGetUserRequest() {
	const responseJson = await sendMyRequest("auth/getme", "GET");
	return responseJson;
}

// 로그인 되었는지 확인 후, 실패했다면 로그인 페이지로 보내버림.
export async function requireLogin() {
	const responseJson = await sendGetUserRequest();
	if(responseJson.status_code !== 200) {
		// 로그인 된 상태가 아니거나 잘못된 요청일 경우
        window.location.href = "login.html"; // 로그인 페이지로 보냄
	}
	return responseJson;
}

// 로그인 되었는지 확인 후, 일반유저가 아니거나 로그인 안되었다면 로그인 페이지로 보내버림
export async function requireNormalUser() {
	const responseJson = await requireLogin();
	if(responseJson.is_superuser){
		// 슈퍼유저면 로그인 페이지로 보냄
        window.location.href = "login.html";
	}
	return responseJson;
}

// 로그인 되었는지 확인 후, 슈퍼유저가 아니거나 로그인 안되었다면 로그인 페이지로 보내버림
export async function requireSuperUser() {
	const responseJson = await requireLogin();
	if(responseJson.is_superuser == false){
		// 슈퍼유저가 아니면 로그인 페이지로 보냄
        window.location.href = "login.html";
	}
	return responseJson;
}

export async function getWebSocket(str_api) {
	const socket = await new WebSocket(BACKEND_WS_URL+str_api);
	return socket
}

// (주)캔버스상에 박스 그리기
// (처음x, 처음y, 끝x, 끝y) 순으로 들어온 float(0~1)값 좌표를 가지고 박스그림
export function drawBoxInCanvas(ctx, canvas, x0, y0, x1, y1) {
    // 실수 : strokeRect는 ymin값이 
    ctx.strokeStyle = 'red';
    ctx.lineWidth = canvas.width / 100;
    const val_x0 = x0 * canvas.width;
    const val_y0 = y0 * canvas.height;
    const tmp_val_x1 = (x1-x0)*canvas.width; 
    const val_x1 = tmp_val_x1 < canvas.width-val_x0 ? tmp_val_x1 : canvas.width-val_x0;
    const tmp_val_y1 = (y1-y0)*canvas.height;
    const val_y1 = tmp_val_y1 < canvas.height-val_y0 ? tmp_val_y1 : canvas.height-val_y0;
    ctx.strokeRect(
        val_x0,
        val_y0,
        val_x1,
        val_y1,
    );
}

// 유저 이름 정규표현식 검사
export function isValidUsername(argUserName) {
	let usernameRegex = /^[a-zA-Z0-9_]{3,20}$/;
	return usernameRegex.test(argUserName);
}

// 비밀번호 정규표현식 검사
export function isValidPassword(argPassword) {
	let passwordRegex = /^(?=.*[a-zA-Z])(?=.*[!@#$%^*+=_-])(?=.*[0-9])[a-zA-Z0-9!@#$%^*+=_-]{7,30}$/;
	return passwordRegex.test(argPassword);
}