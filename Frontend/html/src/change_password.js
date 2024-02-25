import { sendMyRequest, sendLogoutRequest, requireLogin, isValidPassword } from "./common/common_module.js";

pageMain();
// 페이지에서 처음에 바로 실행될 함수
async function pageMain() {
	// 로그인 필요
    const responseJson = await requireLogin();
	document.querySelector("#is_superuser").textContent = responseJson.is_superuser;
	document.querySelector("#username").textContent = responseJson.username;
	document.getElementById('btnchange').addEventListener('click', passwdchange);
}

// 비밀번호 변경 버튼 눌렀을 때 실행될 함수
async function passwdchange() {
	// input 필드에서 값(비밀번호들) 가져오기
	const old_password =  document.getElementById('old_password').value;
	const new_password0 =  document.getElementById('new_password0').value;
	const new_password1 =  document.getElementById('new_password1').value;
	
	// 브라우저 내 입력값 검증
	if ('' in [old_password, new_password0, new_password1]) {
		alert("All three fields must be filled out.")
		return;
	}
	if (new_password0 != new_password1) {
		alert("Password confirmation mismatch.")
		return;
	}

	if (new_password0 == old_password) {
		alert("The new password must be different from the current password!")
		return;
	}

	if (!isValidPassword(new_password0)) {
		alert(
			"The password must consist of English uppercase and lowercase letters, numbers, and special characters, "+
			"with each character represented at least once, "+
			"and must be between 7 and 30 characters long."
		);
		return;
	}

	// 리퀘스트 보내기
	const data = {
		old_password, new_password:new_password0
	};
	
	const responseJson = await sendMyRequest("auth/changepw", "PUT", data);

	// 모종의 이유로 실패시 처리 로직
	if(responseJson.status_code !== 200) {
		alert(responseJson.detail)
		return;
	}

	alert("Password change successful. Please log in again.");
	// 로그아웃 처리
	await sendLogoutRequest();
}


