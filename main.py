"""
카카오톡 채널 AI 민원처리 스킬서버
- 카카오 오픈빌더 스킬(Skill) 연동
- Anthropic Claude API로 자연어 민원 처리
- 긴급 민원 시 임대인에게 알림
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import anthropic
import json
import os
from datetime import datetime
import logging

# ============================================================
# 설정
# ============================================================

# Anthropic API 키 (환경변수로 관리)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")

# 임대인 알림 설정 (긴급 민원 시)
OWNER_NOTIFY_URL = os.getenv("OWNER_NOTIFY_URL", "")  # 카카오톡 나에게 보내기 API 등

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="17호 민원처리 챗봇")

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
    """Claude API로 민원 응답 생성"""
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        ai_text = response.content[0].text
        
        # 긴급 여부 판단
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
# 카카오 오픈빌더 스킬 엔드포인트
# ============================================================

@app.post("/skill/complaint")
async def kakao_skill_complaint(request: Request):
    """
    카카오 오픈빌더 스킬 엔드포인트
    
    오픈빌더에서 이 URL을 스킬로 등록하면,
    입주민의 메시지가 이 서버로 전달되고 AI 응답이 반환됩니다.
    """
    
    body = await request.json()
    logger.info(f"수신된 요청: {json.dumps(body, ensure_ascii=False)}")
    
    # 카카오 오픈빌더 요청에서 사용자 발화 추출
    user_message = body.get("userRequest", {}).get("utterance", "")
    user_id = body.get("userRequest", {}).get("user", {}).get("id", "unknown")
    
    if not user_message:
        return make_kakao_response("무엇을 도와드릴까요? 😊")
    
    # AI 응답 생성
    ai_result = await get_ai_response(user_message, user_id)
    
    # 민원 로그 저장
    log_complaint(user_id, user_message, ai_result["text"], ai_result["is_urgent"])
    
    # 긴급 민원 시 임대인 알림 (별도 구현 필요)
    if ai_result["is_urgent"]:
        logger.warning(f"⚠️ 긴급 민원 발생! 사용자: {user_id}, 내용: {user_message}")
        # TODO: 임대인에게 카카오톡/문자 알림 전송
        # await notify_owner(user_id, user_message)
    
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
    """
    카카오 오픈빌더 스킬 응답 JSON 포맷
    https://i.kakao.com/docs/skill-response-format
    """
    
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
    
    # 바로가기 버튼 추가 (선택)
    if quick_replies:
        response["template"]["quickReplies"] = quick_replies
    else:
        # 기본 바로가기 버튼
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
    """민원 로그 조회 (관리자용)"""
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
    """긴급 민원만 조회"""
    try:
        if os.path.exists(COMPLAINT_LOG_FILE):
            with open(COMPLAINT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
            urgent = [log for log in logs if log.get("is_urgent")]
            return {"total": len(urgent), "logs": urgent}
        return {"total": 0, "logs": []}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
