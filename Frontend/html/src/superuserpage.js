import { requireSuperUser, sendLogoutRequest } from "./common/common_module.js";

pageMain();
async function pageMain() {
	const responseJson = await requireSuperUser(); // 슈퍼 유저만 접근 가능
    
	document.querySelector("#username").textContent = responseJson.username;
	document.querySelector("#passwdchange").addEventListener("click", function() {location.href = "change_password.html";});
	document.querySelector("#passwdreset").addEventListener("click", function() {location.href = "password_reset.html";});
	document.querySelector("#logout").addEventListener("click", sendLogoutRequest);
	document.querySelector("#makeuser").addEventListener("click", function() { location.href = "makeuser.html";});
	document.querySelector("#search").addEventListener("click", function() { location.href = "video_search.html";});
}
