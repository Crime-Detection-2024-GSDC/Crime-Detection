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
		alert("사용자명은 반드시 3글자 이상 20글자 이하의 영어 대소문자, 숫자, 언더바(_)로만 구성되어야 합니다!");
		return;
	}

	// 생성할건지 묻는 창 띄우가
	if(confirm(`새 사용자'${data.username}'를 생성하시겠습니까?`) == false) {
		return;	
	}

	const responseJson = await sendMyRequest("auth/makeuser", "POST", data);
	if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패
		alert(responseJson.detail);
		return;
	}

	alert(`새 사용자 ${responseJson.username}이 생성되었습니다.\n초기 비밀번호 : ${responseJson.password}`)
}