import { requireNormalUser, sendLogoutRequest } from "./common/common_module.js";

pageMain();
async function pageMain() {
	const responseJson = await requireNormalUser(); // 일반 유저만 접근 가능
    document.querySelector("#username").textContent = responseJson.username;
	document.querySelector("#logout").addEventListener("click", sendLogoutRequest);
	document.querySelector("#passwdchange").addEventListener("click", function() {location.href = "change_password.html";});
	document.querySelector("#detect").addEventListener("click", function() {location.href = "send_video.html";});
}
