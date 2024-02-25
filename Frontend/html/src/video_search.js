import { requireSuperUser, sendMyRequest } from "./common/common_module.js";
const userCheckboxList = document.querySelector("#userlist");
const selectDate = document.querySelector("#selectDate");
const selectTime0 = document.querySelector("#selectTime0");
const selectTime1 = document.querySelector("#selectTime1");
const viewCounts = document.querySelector("#viewCounts");
const includeNonViolent = document.querySelector("#includeNonViolent");
const searchTable = document.querySelector("#searchTable");
const prevBtn = document.querySelector("#prevBtn");
const nextBtn = document.querySelector("#nextBtn");
const indexNumber = document.querySelector("#indexNumber");
const searchBtn = document.querySelector("#search");

let body_save = undefined;

pageMain();
pageMainSecond();
async function pageMain() {
    // * userlist에 유저 전부 넣기
    // 유저 목록(슈퍼유저 제외) 받아옴
    const responseJson = await sendMyRequest("auth/getalluser", "GET");
    if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패했다면 알림창 띄우고 중단
		alert(responseJson.detail); return;
	}
    const userlist = responseJson.username_list;
    for(let i=0 ; i < userlist.length; i++) {
        // 추가할 라벨
        let label = document.createElement("label");
        let input = document.createElement("input");
        input.type = "checkbox";
        input.value = userlist[i];
        input.className = "checkbox";
        //input.innerText = userlist[i];
        label.append(input);
        label.innerHTML += userlist[i];
        userCheckboxList.append(label);
	}

    checkOrUncheckAllUser();
}
function pageMainSecond() {
    // 전체 선택 버튼에 메서드 추가
    document.querySelector("#CheckAll").addEventListener("click", checkOrUncheckAllUser);

    // 날짜는 오늘 날짜로
    selectDate.value = getTodayString();

    // 시간은 00:00:00 ~ 23:59:59
    selectTime0.value = "00:00:00";
    selectTime1.value = "23:59:59";

    // 검색 버튼 콜백 지정
    searchBtn.onclick = async () => {
        await search(1, true);
    }

    // 이전/다음버튼 처음에는 안보이게
    indexes.style.display = 'none';
}


// '전체 선택' 버튼 눌렸을 때 실행될 거
function checkOrUncheckAllUser() {
    const userChecks = userCheckboxList.querySelectorAll("input");
    // 하나라도 체크가 안되어있다면 전부 다 체크
    // 전부 다 체크되어있다면 전부 다 체크 안 함
    let all_checked = true;
    for(let i = 0; i < userChecks.length; i++) {
        if(!userChecks[i].checked) {
            all_checked = false;
            break;
        }
    }

    // 하나라도 체크가 안되어있다면 전부 다 체크
    // 전부 다 체크되어있다면 전부 다 체크 안 함
    const to_do = !all_checked
    for(let i = 0; i < userChecks.length; i++) {
        userChecks[i].checked = to_do;
    }
}

// 유저목록에서 체크된 user의 username을 모두 구해옴
function getUsersList() {
    let ret_list = []
    const userChecks = userCheckboxList.querySelectorAll("input");
    for(let i = 0; i < userChecks.length; i++) {
        if(userChecks[i].checked) {
            ret_list.push(userChecks[i].value);
        }
    }
    return ret_list;
}

// 검색창 함수
async function search(val_index, isSearchBtn) {
    disableAllBtns();
    let responseJson = null;
    //indexes.style.display = 'none';
    if(isSearchBtn) {
        // '검색' 버튼을 눌러서 실행되었을 경우
        // 검색 요청 보내기
        const body = {
            "usernames" : getUsersList(),
            "date" : selectDate.value,
            "timeStart" : selectTime0.value,
            "timeEnd" : selectTime1.value,
            "index" : val_index,
            "viewCounts" : Number(viewCounts.value),
            "searchAll" : includeNonViolent.checked
        }
        // 인덱스 눌렀을 때도 쓸 수 있게 놔둠
        body_save = body;
        responseJson = await sendMyRequest("infer/searchpage", "POST", body);
    }
    else {
        body_save['index'] = val_index;
        responseJson = await sendMyRequest("infer/searchpage", "POST", body_save);
    }
    if(responseJson.status_code !== 200) {
		// 모종의 이유로 실패했다면 알림창 띄우고 중단
		alert(responseJson.detail);
        //indexes.style.display = 'block'; 
        undisableAllBtns();
        return;
	}
    // 검색 테이블이랑 인덱스 버튼 비움
    clearTable();
    // 이후 받아온 검색 데이터들을 가지고 이것저것 만짐
    const totalIndex = responseJson.totalIndex;
    const results = responseJson.results;

    // 테이블에 추가
    for(let i = 0; i < results.length; i++) {
        // 추가할 tr태그
        let tr = document.createElement("tr");
        const responseObj = results[i];
        const id = responseObj.id;
        
        let idField = document.createElement("td");
        idField.innerText = id;
        tr.append(idField);
        
        let usernameField = document.createElement("td");
        usernameField.innerText = responseObj.username;
        tr.append(usernameField);

        let dateField = document.createElement("td");
        dateField.innerText = responseObj.date;
        tr.append(dateField);

        let isViolentField = document.createElement("td");
        isViolentField.innerText = responseObj.isViolent ? "O" : "X";
        tr.append(isViolentField);

        tr.className = "searchElement";
        tr.onclick = async () => {await openVideoShow(id);};
        searchTable.append(tr);
    }
    undisableAllBtns();

    indexNumber.innerText = val_index + "/" + totalIndex;
    indexNumber.onclick = async () => {
        const input_num = prompt("Enter page number (1 ~ " + totalIndex + ")");
        if(input_num===undefined) { return; } // 입력 안하고 X키 누르면 그냥 창 닫기
        const numbered_input_num = Number(input_num);
        if(isNaN(numbered_input_num) || (!Number.isInteger(numbered_input_num))) {
            // 정수가 아닌 무언가를 입력했다면
            alert("Only integer input is allowed.");
           return;
        }
        if(numbered_input_num < 1 || numbered_input_num > totalIndex) {
            // 범위 초과시
            alert("Exceeded allowable range. (1 ~ " + totalIndex + ")"); 
            return;
        }
        await search(numbered_input_num, false);
    }

    if(val_index > 1) {
        prevBtn.onclick = async () => { await search(val_index-1, false); }
    }
    else {
        prevBtn.onclick = () => { alert("This is the first page!"); };
    }

    if(val_index < totalIndex) {
        nextBtn.onclick = async () => { await search(val_index+1, false); }
    }
    else {
        nextBtn.onclick = () => { alert("This is the last page!"); };
    }
    indexes.style.display = 'block';
}


function getTodayString() {
    const today = new Date();
    const year = today.getFullYear();
    const month = ('0' + (today.getMonth() + 1)).slice(-2);
    const day = ('0' + today.getDate()).slice(-2);
    const dateString = year + '-' + month  + '-' + day;
    return dateString;
}

// 검색결과 테이블을 비우기
function clearTable() {
    const length = searchTable.querySelectorAll("tr").length
    for(let i=0; i < length; i++) {
        searchTable.deleteRow(-1);
    }
}

async function openVideoShow(val_id) {
    localStorage.setItem("videoPK", Number(val_id));
    window.open("/view_video.html", "_blank");
}

function disableAllBtns() {
    prevBtn.disabled = true;
    nextBtn.disabled = true;
}

function undisableAllBtns() {
    prevBtn.disabled = false;
    nextBtn.disabled = false;
}