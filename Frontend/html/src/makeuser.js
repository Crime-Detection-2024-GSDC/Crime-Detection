import { sendMyRequest, requireSuperUser, isValidUsername } from "./common/common_module.js";

pageMain();
async function pageMain() {
	await requireSuperUser();  // 슈퍼유저만 접근 가능
	document.querySelector("#btn_make").addEventListener("click", makeUser);
}

// 비밀번호 변경 버튼 눌렀을 때 실행될 함수
async function makeUser() {
	const data = {username : document.querySelector("#username").value}
	
	if(!isValidUsername(data.username)) {
		alert("Username must consist of 3 to 20 characters, including English uppercase and lowercase letters, numbers, and underscores(_).");
		return;
	}

	// 생성할건지 묻는 창 띄우가
	if (confirm(`Create a new user '${data.username}'?`) == false) {
		return;	
	}

	const responseJson = await sendMyRequest("auth/makeuser", "POST", data);
	if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패
		alert(responseJson.detail);
		return;
	}

	alert(`New user ${responseJson.username} has been created.\nInitial password: ${responseJson.password}`);
}