import { requireSuperUser, sendMyRequest } from "./common/common_module.js";

pageMain();
async function pageMain() {
	await requireSuperUser(); //슈퍼유저만 접근 가능
	await putAllUsersInSelect();
	document.querySelector("#btnreset").addEventListener("click", resetPw);
}

// select에 유저 리스트 전부 넣기
async function putAllUsersInSelect() {
	const responseJson = await sendMyRequest("auth/getalluser", "GET");

	if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패
		alert(responseJson.detail);
		return;
	}
	let select = document.querySelector("#userlist");
	const userlist = responseJson.username_list;
	
	for(let i=0 ; i < userlist.length; i++) {
		let option = document.createElement("option");
		option.innerText = userlist[i];
		select.append(option);
	}
}

// 초기화 버튼 눌렸을 때 실행될 함수
async function resetPw() {
	const username = document.querySelector("#userlist").value;
	// 초기화할건지 묻는 창 띄우기
	if (confirm(`Reset the password for user '${username}'?`) == false) {
		return;	
	}
	const data = { username };
	const responseJson = await sendMyRequest("auth/resetpw", "POST", data);
	if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패
		alert(responseJson.detail);
		return;
	}
	alert(`Password for user '${username}' has been reset.\nNew password: ${responseJson.new_password}`);
}