"""
카카오톡 채널 AI 민원처리 스킬서버 (콜백 방식)
- 즉시 "확인했습니다" 응답 → 백그라운드에서 AI 처리 → 콜백으로 실제 답변 전송
- 5초 타임아웃 문제 완전 해결
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import anthropic
import httpx
import json
import os
import asyncio
from datetime import datetime
import logging

# ============================================================
# 설정
# ============================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
OWNER_NOTIFY_URL = os.getenv("OWNER_NOTIFY_URL", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="17호 민원처리 챗봇")

# ============================================================
# 유저별 대화 기억 (최근 5턴 저장)
# ============================================================

CHAT_HISTORY_FILE = "chat_history.json"
MAX_AI_CONTEXT = 5  # AI에게 보내는 최근 대화 수 (비용/속도 관리)


def load_chat_history() -> dict:
    """전체 대화 기록 로드"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"대화 기록 로드 실패: {e}")
    return {}


def save_chat_history(history: dict):
    """전체 대화 기록 저장"""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"대화 기록 저장 실패: {e}")


def get_user_messages(user_id: str) -> list:
    """유저의 최근 대화를 Claude API 형식으로 반환 (최근 MAX_AI_CONTEXT턴만)"""
    history = load_chat_history()
    user_history = history.get(user_id, [])
    
    # 최근 N턴만 AI에게 전달 (전체는 보관)
    recent = user_history[-MAX_AI_CONTEXT:]
    
    messages = []
    for turn in recent:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    
    return messages


def add_to_history(user_id: str, user_message: str, ai_response: str):
    """유저 대화 기록에 새 턴 추가 (전체 보관, 삭제 안 함)"""
    history = load_chat_history()
    
    if user_id not in history:
        history[user_id] = []
    
    history[user_id].append({
        "user": user_message,
        "assistant": ai_response,
        "timestamp": datetime.now().isoformat()
    })
    
    save_chat_history(history)


# ============================================================
# 건물 정보 & 민원 지식베이스
# ============================================================

# ============================================================
# 학습 데이터 로드 (knowledge.json)
# ============================================================

KNOWLEDGE_FILE = "knowledge.json"


def load_knowledge() -> str:
    """knowledge.json에서 학습 데이터를 읽어 텍스트로 변환"""
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _format_knowledge(data)
    except Exception as e:
        logger.error(f"학습 데이터 로드 실패: {e}")
    return "(등록된 건물 정보가 없습니다)"


def _format_knowledge(data: dict, indent: int = 0) -> str:
    """중첩 JSON을 읽기 좋은 텍스트로 변환"""
    lines = []
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}[{key}]")
            lines.append(_format_knowledge(value, indent + 1))
        else:
            lines.append(f"{prefix}- {key}: {value}")
    return "\n".join(lines)

# ============================================================
# 봇 일시정지 관리 (직접 상담 모드)
# ============================================================

PAUSED_USERS_FILE = "paused_users.json"


def load_paused_users() -> dict:
    """일시정지된 유저 목록 로드"""
    try:
        if os.path.exists(PAUSED_USERS_FILE):
            with open(PAUSED_USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"일시정지 목록 로드 실패: {e}")
    return {}


def save_paused_users(paused: dict):
    """일시정지된 유저 목록 저장"""
    try:
        with open(PAUSED_USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(paused, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"일시정지 목록 저장 실패: {e}")


def is_user_paused(user_id: str) -> bool:
    """유저가 일시정지(직접상담 모드)인지 확인"""
    paused = load_paused_users()
    return user_id in paused


def pause_user(user_id: str):
    """유저 봇 일시정지 (직접상담 모드 전환)"""
    paused = load_paused_users()
    paused[user_id] = {"paused_at": datetime.now().isoformat()}
    save_paused_users(paused)


def resume_user(user_id: str):
    """유저 봇 다시 활성화"""
    paused = load_paused_users()
    if user_id in paused:
        del paused[user_id]
        save_paused_users(paused)

# ============================================================
# Claude AI 응답 생성
# ============================================================

def get_system_prompt() -> str:
    """시스템 프롬프트 생성 (매번 최신 knowledge.json 반영)"""
    knowledge = load_knowledge()
    return f"""당신은 다가구주택 건물 관리 AI 도우미입니다.
입주민의 민원과 질문을 접수하고 대응합니다.

## ⚠️ 절대 원칙 (반드시 지키세요)
1. 확실하지 않은 정보는 절대 말하지 마세요
2. 건물 정보에 없는 내용은 추측하지 마세요
3. 수리비, 보상, 계약 조건 등 책임이 따르는 답변은 하지 마세요
4. 모르는 것은 "관리자에게 확인 후 안내드리겠습니다"로 답하세요
5. 당신은 AI 도우미일 뿐이며, 최종 결정권은 임대인(관리자)에게 있음을 명시하세요

## 대응 원칙

### 1단계: 분류
입주민 메시지를 아래 중 하나로 분류하세요:
- 긴급: 누수, 화재, 가스, 정전, 침입 등 즉시 조치 필요
- 시설: 보일러, 수도, 전기, 엘리베이터 등 시설물 문제
- 생활: 소음, 주차, 쓰레기, 벌레 등 생활 불편
- 문의: 관리비, 계약, 일정 등 정보 요청
- 기타: 위에 해당하지 않는 것

### 2단계: 대응
- 긴급 → [긴급] 태그 + 안전 확보 안내 + 임대인 연락 안내
- 시설 → 자가 점검 방법 안내 → 안 되면 "관리자에게 전달하겠습니다" 안내
- 생활 → 해결 방법 안내 → 필요 시 "관리자에게 전달하겠습니다" 안내
- 문의 → 건물 정보에 있으면 답변, 없으면 "관리자에게 확인 후 안내드리겠습니다"
- 기타 → 최대한 도움 제공, 모르면 "관리자에게 전달하겠습니다"

### 3단계: 후속 확인
- 문제 해결 여부를 물어보세요
- 추가로 필요한 것이 있는지 확인하세요

## 답변 규칙
- 카카오톡 메시지이므로 간결하게 (최대 300자)
- 이모지 적절히 활용
- 존댓말 사용
- 이전 대화 맥락을 반영하여 답변

## 긴급 판단 키워드
누수, 물이 새, 침수, 화재, 불, 연기, 가스 냄새, 가스 누출, 정전, 문 안 열림, 잠김, 도둑, 침입

## 건물 정보
{knowledge}
"""


async def get_ai_response(user_message: str, user_id: str = "") -> dict:
    """Claude API로 민원 응답 생성 (이전 대화 기억 포함)"""
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    try:
        # 이전 대화 불러오기
        previous_messages = get_user_messages(user_id)
        
        # 이전 대화 + 새 메시지
        messages = previous_messages + [{"role": "user", "content": user_message}]
        
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=get_system_prompt(),
            messages=messages
        )
        
        ai_text = response.content[0].text
        
        # 대화 기록 저장
        add_to_history(user_id, user_message, ai_text)
        
        is_urgent = "[긴급]" in ai_text or any(
            keyword in user_message 
            for keyword in ["누수", "물이 새", "침수", "화재", "불이", "연기", "가스", "정전"]
        )
        
        return {
            "text": ai_text,
            "is_urgent": is_urgent,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI 응답 생성 실패: {e}")
        return {
            "text": "죄송합니다, 일시적인 오류가 발생했습니다. 😅\n긴급한 문의는 임대인에게 직접 연락해 주세요.",
            "is_urgent": False,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }


# ============================================================
# 민원 로그 저장
# ============================================================

COMPLAINT_LOG_FILE = "complaint_log.json"


def log_complaint(user_id: str, message: str, response: str, is_urgent: bool):
    """민원 내역을 JSON 파일에 저장"""
    try:
        if os.path.exists(COMPLAINT_LOG_FILE):
            with open(COMPLAINT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "message": message,
            "response": response,
            "is_urgent": is_urgent,
            "status": "접수" if is_urgent else "자동처리"
        })
        
        with open(COMPLAINT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"로그 저장 실패: {e}")


# ============================================================
# 콜백으로 AI 응답 전송 (백그라운드)
# ============================================================

async def process_and_callback(callback_url: str, user_message: str, user_id: str):
    """백그라운드에서 AI 응답 생성 후 카카오 콜백으로 전송"""
    try:
        # AI 응답 생성 (시간 제한 없음)
        ai_result = await get_ai_response(user_message, user_id)
        
        # 민원 로그 저장
        log_complaint(user_id, user_message, ai_result["text"], ai_result["is_urgent"])
        
        # 긴급 민원 알림
        if ai_result["is_urgent"]:
            logger.warning(f"⚠️ 긴급 민원 발생! 사용자: {user_id}, 내용: {user_message}")
        
        # 콜백 응답 포맷
        callback_response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": ai_result["text"]
                        }
                    }
                ]
            }
        }
        
        # 카카오 콜백 URL로 응답 전송
        async with httpx.AsyncClient() as http_client:
            result = await http_client.post(
                callback_url,
                json=callback_response,
                timeout=10.0
            )
            logger.info(f"콜백 전송 완료: {result.status_code}")
            
    except Exception as e:
        logger.error(f"콜백 처리 실패: {e}")


# ============================================================
# 카카오 오픈빌더 스킬 엔드포인트 (콜백 방식)
# ============================================================

@app.post("/skill/complaint")
async def kakao_skill_complaint(request: Request):
    """
    카카오 오픈빌더 스킬 엔드포인트 (콜백 방식)
    
    1. 즉시 "확인했습니다" 응답 반환 (1초 이내)
    2. 백그라운드에서 AI 처리
    3. 콜백 URL로 실제 답변 전송
    """
    
    body = await request.json()
    logger.info(f"수신된 요청: {json.dumps(body, ensure_ascii=False)}")
    
    user_message = body.get("userRequest", {}).get("utterance", "")
    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown")
    callback_url = body.get("userRequest", {}).get("callbackUrl", "")
    
    if not user_message:
        return make_kakao_response("무엇을 도와드릴까요? 😊")
    
    # 봇 일시정지 상태면 → 완전 무응답 (관리자가 직접 상담 중)
    if is_user_paused(user_id):
        logger.info(f"봇 일시정지 중 - 유저: {user_id}, 메시지: {user_message}")
        # 의도적으로 지연시켜 타임아웃 유도 → 카카오가 아무 메시지도 안 보냄
        await asyncio.sleep(6)
        return JSONResponse(content={"version": "2.0", "template": {"outputs": []}})
    
    # 콜백 URL이 있으면 → 콜백 방식 (즉시 응답 + 백그라운드 처리)
    if callback_url:
        # 백그라운드에서 AI 처리 시작
        asyncio.create_task(process_and_callback(callback_url, user_message, user_id))
        
        # 즉시 응답 반환 (useCallback: true)
        return JSONResponse(content={
            "version": "2.0",
            "useCallback": True,
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "확인했습니다! 잠시만 기다려 주세요 😊"
                        }
                    }
                ]
            }
        })
    
    # 콜백 URL이 없으면 → 직접 응답 (기존 방식)
    ai_result = await get_ai_response(user_message, user_id)
    log_complaint(user_id, user_message, ai_result["text"], ai_result["is_urgent"])
    
    if ai_result["is_urgent"]:
        logger.warning(f"⚠️ 긴급 민원 발생! 사용자: {user_id}, 내용: {user_message}")
    
    return make_kakao_response(ai_result["text"])


@app.post("/skill/info")
async def kakao_skill_info(request: Request):
    """건물 기본 정보 안내 스킬"""
    info_text = """🏠 건물 관리 도우미입니다.

💬 궁금한 점은 편하게 물어보세요! 😊"""
    return make_kakao_response(info_text)


@app.post("/skill/emergency")
async def kakao_skill_emergency(request: Request):
    """긴급 연락처 안내 스킬"""
    emergency_text = """🚨 긴급 연락처

🔥 화재/응급: 119
🚔 범죄/소음: 112
💧 수도 긴급: 120
⛽ 가스 긴급: 1588-5788"""
    return make_kakao_response(emergency_text)


# ============================================================
# 카카오 오픈빌더 응답 포맷
# ============================================================

def make_kakao_response(text: str, quick_replies: list = None):
    response = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": text
                    }
                }
            ]
        }
    }
    
    # 버튼이 명시적으로 전달된 경우에만 추가
    if quick_replies:
        response["template"]["quickReplies"] = quick_replies
    
    return JSONResponse(content=response)


# ============================================================
# 관리자 웹 페이지 (핸드폰에서 접속)
# ============================================================

ADMIN_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>민원 챗봇 관리</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 16px; }
        h1 { font-size: 20px; margin-bottom: 16px; color: #333; }
        h2 { font-size: 16px; margin: 20px 0 10px; color: #555; }
        .card { background: white; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .user-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #eee; }
        .user-row:last-child { border-bottom: none; }
        .user-id { font-size: 13px; color: #666; word-break: break-all; flex: 1; margin-right: 10px; }
        .user-last { font-size: 11px; color: #999; }
        .btn { padding: 8px 16px; border: none; border-radius: 8px; font-size: 14px; font-weight: bold; cursor: pointer; min-width: 70px; }
        .btn-pause { background: #ff6b6b; color: white; }
        .btn-resume { background: #51cf66; color: white; }
        .btn-pause:active { background: #e55a5a; }
        .btn-resume:active { background: #40c057; }
        .status { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
        .status-bot { background: #d3f9d8; color: #2b8a3e; }
        .status-human { background: #ffe3e3; color: #c92a2a; }
        .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }
        .stat-box { background: white; border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stat-num { font-size: 28px; font-weight: bold; color: #333; }
        .stat-label { font-size: 12px; color: #888; margin-top: 4px; }
        .empty { color: #999; text-align: center; padding: 20px; font-size: 14px; }
        .refresh-btn { display: block; width: 100%; padding: 12px; background: #228be6; color: white; border: none; border-radius: 12px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 16px; }
    </style>
</head>
<body>
    <h1>🏠 민원 챗봇 관리</h1>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-num" id="totalUsers">-</div>
            <div class="stat-label">전체 유저</div>
        </div>
        <div class="stat-box">
            <div class="stat-num" id="pausedCount">-</div>
            <div class="stat-label">직접상담 중</div>
        </div>
    </div>

    <h2>💬 직접상담 중 (봇 꺼짐)</h2>
    <div class="card" id="pausedList">
        <div class="empty">직접상담 중인 유저가 없습니다</div>
    </div>

    <h2>🤖 봇 활성 유저</h2>
    <div class="card" id="activeList">
        <div class="empty">로딩 중...</div>
    </div>

    <button class="refresh-btn" onclick="loadData()">🔄 새로고침</button>

    <script>
        async function loadData() {
            try {
                const [historyRes, pausedRes] = await Promise.all([
                    fetch('/admin/history'),
                    fetch('/admin/paused')
                ]);
                const history = await historyRes.json();
                const paused = await pausedRes.json();
                
                const pausedIds = new Set(Object.keys(paused.paused_users || {}));
                const users = history.users || {};
                
                document.getElementById('totalUsers').textContent = history.total_users || 0;
                document.getElementById('pausedCount').textContent = pausedIds.size;
                
                // 직접상담 중 목록
                let pausedHtml = '';
                for (const [uid, info] of Object.entries(users)) {
                    if (pausedIds.has(uid)) {
                        const shortId = uid.substring(0, 12) + '...';
                        pausedHtml += `
                            <div class="user-row">
                                <div>
                                    <div class="user-id">${shortId}</div>
                                    <div class="user-last">대화 ${info.total_turns}건</div>
                                    <span class="status status-human">직접상담</span>
                                </div>
                                <button class="btn btn-resume" onclick="resumeBot('${uid}')">봇 켜기</button>
                            </div>`;
                    }
                }
                // paused에 있지만 history에 없는 유저도 표시
                for (const uid of pausedIds) {
                    if (!users[uid]) {
                        const shortId = uid.substring(0, 12) + '...';
                        pausedHtml += `
                            <div class="user-row">
                                <div>
                                    <div class="user-id">${shortId}</div>
                                    <span class="status status-human">직접상담</span>
                                </div>
                                <button class="btn btn-resume" onclick="resumeBot('${uid}')">봇 켜기</button>
                            </div>`;
                    }
                }
                document.getElementById('pausedList').innerHTML = pausedHtml || '<div class="empty">직접상담 중인 유저가 없습니다</div>';
                
                // 봇 활성 목록
                let activeHtml = '';
                for (const [uid, info] of Object.entries(users)) {
                    if (!pausedIds.has(uid)) {
                        const shortId = uid.substring(0, 12) + '...';
                        const lastTime = info.last ? new Date(info.last).toLocaleString('ko-KR') : '';
                        activeHtml += `
                            <div class="user-row">
                                <div>
                                    <div class="user-id">${shortId}</div>
                                    <div class="user-last">${lastTime} · ${info.total_turns}건</div>
                                    <span class="status status-bot">봇 활성</span>
                                </div>
                                <button class="btn btn-pause" onclick="pauseBot('${uid}')">상담</button>
                            </div>`;
                    }
                }
                document.getElementById('activeList').innerHTML = activeHtml || '<div class="empty">활성 유저가 없습니다</div>';
                
            } catch (e) {
                console.error(e);
            }
        }
        
        async function pauseBot(userId) {
            if (!confirm('이 유저의 봇을 끄고 직접 상담하시겠습니까?')) return;
            await fetch('/admin/pause/' + encodeURIComponent(userId), { method: 'POST' });
            loadData();
        }
        
        async function resumeBot(userId) {
            if (!confirm('이 유저의 봇을 다시 켜시겠습니까?')) return;
            await fetch('/admin/resume/' + encodeURIComponent(userId), { method: 'POST' });
            loadData();
        }
        
        loadData();
    </script>
</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """관리자 웹 페이지"""
    return ADMIN_HTML


@app.get("/admin/paused")
async def get_paused_users():
    """일시정지된 유저 목록 조회"""
    return {"paused_users": load_paused_users()}


@app.post("/admin/pause/{user_id}")
async def pause_user_bot(user_id: str):
    """유저 봇 일시정지 (직접상담 모드)"""
    pause_user(user_id)
    logger.info(f"🔴 봇 일시정지: {user_id}")
    return {"message": f"봇 일시정지 완료 - 직접상담 모드", "user_id": user_id}


@app.post("/admin/resume/{user_id}")
async def resume_user_bot(user_id: str):
    """유저 봇 다시 활성화"""
    resume_user(user_id)
    logger.info(f"🟢 봇 재활성화: {user_id}")
    return {"message": f"봇 재활성화 완료", "user_id": user_id}


# ============================================================
# 관리자용 데이터 엔드포인트
# ============================================================

@app.get("/admin/knowledge")
async def get_knowledge():
    """현재 학습 데이터 조회"""
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"message": "knowledge.json 파일 없음"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/admin/logs")
async def get_complaint_logs():
    try:
        if os.path.exists(COMPLAINT_LOG_FILE):
            with open(COMPLAINT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
            return {"total": len(logs), "logs": logs}
        return {"total": 0, "logs": []}
    except Exception as e:
        return {"error": str(e)}


@app.get("/admin/urgent")
async def get_urgent_complaints():
    try:
        if os.path.exists(COMPLAINT_LOG_FILE):
            with open(COMPLAINT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
            urgent = [log for log in logs if log.get("is_urgent")]
            return {"total": len(urgent), "logs": urgent}
        return {"total": 0, "logs": []}
    except Exception as e:
        return {"error": str(e)}


@app.get("/admin/history")
async def get_chat_history():
    """전체 유저 대화 기록 조회 (요약)"""
    history = load_chat_history()
    summary = {
        uid: {"total_turns": len(turns), "first": turns[0]["timestamp"] if turns else "", "last": turns[-1]["timestamp"] if turns else ""}
        for uid, turns in history.items()
    }
    return {"total_users": len(history), "users": summary}


@app.get("/admin/history/{user_id}")
async def get_user_chat_history(user_id: str):
    """특정 유저 전체 대화 기록 조회"""
    history = load_chat_history()
    user_history = history.get(user_id, [])
    return {"user_id": user_id, "total_turns": len(user_history), "history": user_history}


@app.delete("/admin/history/{user_id}")
async def clear_user_chat_history(user_id: str):
    """퇴실 시 유저 대화 기록 삭제"""
    history = load_chat_history()
    if user_id in history:
        del history[user_id]
        save_chat_history(history)
        return {"message": f"유저 {user_id} 대화 기록 삭제 완료"}
    return {"message": "해당 유저 기록 없음"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
