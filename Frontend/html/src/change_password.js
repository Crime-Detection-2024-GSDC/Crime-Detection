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
		alert("세 필드 모두 작성해야 합니다.")
		return;
	}
	if (new_password0 != new_password1) {
		alert("비밀번호 확인이 틀렸습니다.")
		return;
	}

	if (new_password0 == old_password) {
		alert("변경하려는 비밀번호는 기존 비밀번호와 달라야 합니다!")
		return;
	}

	if (!isValidPassword(new_password0)) {
		alert(
			"비밀번호는 반드시 영어 대소문자, 숫자, 특수문자로만 구성되어야 하고, "+
			"각 문자를 1글자 이상 포함하여야 하며, "+
			"7글자 이상 30글자 이하여야합니다."
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

	alert(`비밀번호 변경 성공. 다시 로그인 해 주세요.`);
	// 로그아웃 처리
	await sendLogoutRequest();
}


