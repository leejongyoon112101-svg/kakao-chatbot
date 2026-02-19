"""
카카오톡 채널 AI 민원처리 스킬서버 (콜백 방식)
- 즉시 "확인했습니다" 응답 → 백그라운드에서 AI 처리 → 콜백으로 실제 답변 전송
- 5초 타임아웃 문제 완전 해결
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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

BUILDING_KNOWLEDGE = """
(여기에 건물 정보를 추가하세요)
"""

# ============================================================
# Claude AI 응답 생성
# ============================================================

SYSTEM_PROMPT = f"""당신은 건물 관리 AI 도우미입니다.
입주민의 민원과 질문에 친절하고 실용적으로 답변합니다.

## 답변 규칙
1. 짧고 명확하게 답변 (카카오톡 메시지이므로 간결하게, 최대 300자)
2. 이모지를 적절히 활용
3. 자가 해결 가능하면 단계별 안내
4. 긴급 상황이면 [긴급] 태그를 붙이고 임대인 연락 안내
5. 등록된 정보가 없는 내용은 임대인에게 문의하라고 안내
6. 존댓말 사용

## 긴급 상황 판단 기준
다음 키워드가 포함되면 긴급으로 분류:
- 누수, 물이 새, 침수, 화재, 불, 연기, 가스 냄새, 가스 누출
- 정전, 문 안 열림, 잠김, 도둑, 침입

## 건물 정보
{BUILDING_KNOWLEDGE}
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
            system=SYSTEM_PROMPT,
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
                ],
                "quickReplies": [
                    {
                        "messageText": "긴급 연락처",
                        "action": "message",
                        "label": "🚨 긴급연락처"
                    },
                    {
                        "messageText": "건물 안내",
                        "action": "message",
                        "label": "🏠 건물안내"
                    },
                    {
                        "messageText": "보일러 문제",
                        "action": "message",
                        "label": "🔧 보일러"
                    },
                    {
                        "messageText": "수도 문제",
                        "action": "message",
                        "label": "💧 수도"
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
    
    if quick_replies:
        response["template"]["quickReplies"] = quick_replies
    else:
        response["template"]["quickReplies"] = [
            {
                "messageText": "긴급 연락처",
                "action": "message",
                "label": "🚨 긴급연락처"
            },
            {
                "messageText": "건물 안내",
                "action": "message",
                "label": "🏠 건물안내"
            },
            {
                "messageText": "보일러 문제",
                "action": "message",
                "label": "🔧 보일러"
            },
            {
                "messageText": "수도 문제",
                "action": "message",
                "label": "💧 수도"
            }
        ]
    
    return JSONResponse(content=response)


# ============================================================
# 관리자용 엔드포인트
# ============================================================

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
