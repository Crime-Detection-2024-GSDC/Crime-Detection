import { sendGetUserRequest, sendMyRequest } from "./common/common_module.js";

pageMain();
async function pageMain() {	
	// 이미 로그인 된 상태라면 이동시키기
	const responseJson = await sendGetUserRequest();
	if(responseJson.status_code == 200) {
		location.href = responseJson.is_superuser ? "superuserpage.html" : "normaluserpage.html";
		return;
	}

	document.getElementById('loginButton').addEventListener('click', login);
	document.getElementById('password').addEventListener('keydown', async function(e){if(e.key == "Enter") {await login();}});
}


// 로그인 버튼 눌렀을 때 실행될 함수
async function login() {
	const data = {
		username: document.getElementById('username').value,
		password: document.getElementById('password').value
	};

	const responseJson = await sendMyRequest("auth/login", "POST", data);

	if(responseJson.status_code !== 200) {
		// 로그인 실패
		alert(responseJson.detail)
	}
	else {
		alert(`로그인 성공. 환영합니다 ${responseJson.username}`);
		location.href = responseJson.is_superuser ? "superuserpage.html" : "normaluserpage.html";
	}
}
