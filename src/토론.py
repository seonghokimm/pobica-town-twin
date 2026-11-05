"""
화순군 100인 합성 시민 토론 및 정책 반응 보고서 초안.

이 스크립트는 100명 정책 반응 생성과 군집화가 끝난 뒤 실행하는 단계입니다.
같은 TT 폴더 안에 아래 파일들이 있다고 가정합니다.

    hwasun_persona_policy_responses.json
    outputs/<군집화_실험_폴더>/representative_agents.json

기본 실행:
    python 토론.py --api-mode api

GPT-4.1 일반 모델로 실행(기본값):
    python 토론.py --api-mode api --model gpt-4.1

특정 군집화 결과를 지정해서 실행:
    python 화순_토론_보고서_초안_0705.py --cluster-dir outputs/jobcenter_prompt_v2_hdbscan_20260705_175626

API 없이 로컬 규칙 기반으로만 테스트:
    python 화순_토론_보고서_초안_0705.py --no-api

출력:
    outputs/hwasun_debate_report_<timestamp>/
        debate_result.json
        policy_reaction_report.md
        debate_transcript.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from datetime import datetime
from pathlib import Path
from typing import Any


UTF8_SIG = "utf-8-sig"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESPONSE_PATH = SCRIPT_DIR / "hwasun_persona_policy_responses_d3.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
DEFAULT_RAG_DB_PATH = SCRIPT_DIR / "town_twin_full_v3_policy_stage.sqlite3"
UNVERIFIED_TEXT = "토론 기록상 확인되지 않음"
ANSWER_TYPE_ENUM = ("입장고수", "조건부수용", "부분수용")
FOLLOWUP_ACCEPT_ENUM = ("일부해소", "미해소")

# Debate-rule rationale:
# - Toulmin(1958): opening/rebuttal fields are split into claim-ground-warrant.
# - van Eemeren & Grootendorst(2004): rebuttals must address a concrete opposing point.
# - Fishkin(2009): the moderator facilitates deliberation, not policy invention.
# - Liang et al.(2023, DoT): generate-then-verify keeps multi-turn debate grounded.
# - Sharma et al.(2023, sycophancy): prompts frame interests/losses instead of asking agents to agree.

SCORE_LABELS = {
    1: "매우 불만족",
    2: "불만족",
    3: "보통/조건부",
    4: "만족",
    5: "매우 만족",
}

DEFAULT_TARGET_CSV_PATH = Path("/home/piai/다운로드/교육환경만족도-공교육환경만족도 - 시트1.csv")
DEFAULT_TARGET_CSV_LABEL = "2025년"
TARGET_CSV_SCORE_COLUMNS = {
    1: "매우불만족",
    2: "약간불만족",
    3: "보통",
    4: "약간만족",
    5: "매우만족",
}

RESOURCE_TAG_RULES = {
    "이용권운영": ("이용권", "쿠폰", "월 2매", "월 2장", "월 4장", "지급 매수", "지급량", "이용 횟수", "사용기한", "소멸", "이월"),
    "기사참여": ("택시기사", "택시 기사", "기사 호출", "운전자", "호출 거부", "호출 기피", "배차 거부", "호출 수락", "참여 기사", "참여 차량"),
    "결제정산": ("정산", "환전", "영수증", "거리 기록", "운행 기록", "종이쿠폰", "종이 쿠폰", "카드결제", "카드 결제", "교통카드"),
    "동승운영": ("동승", "합승", "같이 타", "함께 타", "공동 배차", "동승 인원"),
    "재정지속성": ("국비", "도비", "군비", "재정", "지속가능", "지속 가능", "예산 부담", "재원"),
    "생활이동": ("병원", "진료", "약국", "시장", "장날", "장보기", "면사무소", "군청", "행정기관"),
    "택시산업": ("택시업계", "택시 업계", "택시사업자", "운행 수요", "업계 활성화", "운송 수입"),
    "접근성": ("접근", "거리", "멀", "읍내", "이동", "버스", "교통", "온라인", "출장", "가기", "가려", "불편", "시간", "어렵"),
    "대상적합성": ("대상", "나이", "연령", "노인", "고령", "청년", "중장년", "경력단절", "취약"),
    "취업지원": ("취업", "일자리", "구직", "채용", "상담", "훈련", "자격증", "박람회", "알선"),
    "비용형평성": ("예산", "비용", "형평", "소외", "우선순위", "사각지대", "공정", "세금", "낭비", "부담"),
    "운영품질": ("전문", "연계", "정보", "프로그램", "운영", "맞춤", "기관", "기업"),
    "디지털": ("온라인", "디지털", "스마트폰", "영상", "플랫폼", "비대면"),
    "교통": ("버스", "노선", "정류장", "배차", "주차", "택시", "셔틀", "도로", "보행", "대중교통"),
    "보건의료": ("병원", "의료", "진료", "건강", "검진", "의사", "간호", "보건소", "응급", "치료", "약국"),
    "농업": ("농업", "농가", "농민", "농촌", "작물", "농기계", "귀농", "판로", "영농", "축산", "재배"),
    "관광문화": ("관광", "축제", "문화", "행사", "체험", "방문객", "숙박", "전시", "공연", "체육"),
    "교육": ("교육", "학교", "학생", "청소년", "강좌", "학습", "수업", "돌봄교실", "평생학습", "도서관"),
    "돌봄복지": ("돌봄", "복지", "장애인", "노인", "아동", "요양", "취약계층", "저소득", "보호", "가구"),
    "지역경제": ("상권", "소상공인", "시장", "매출", "창업", "기업", "산업", "고용", "소득", "경제"),
    "환경": ("환경", "쓰레기", "재활용", "하천", "미세먼지", "소음", "탄소", "녹지", "공원", "수질"),
    "주거정주": ("주거", "주택", "임대", "빈집", "정주", "마을", "생활권", "인구", "청년주택"),
    "안전재난": ("안전", "재난", "침수", "화재", "범죄", "방범", "CCTV", "폭염", "한파", "위험"),
}

POLICY_CATEGORY_RULES = {
    "공교육": ("공교육", "교육", "학교", "학생", "학습", "방과후", "장학", "학자금", "입학", "청소년"),
    "보육": ("보육", "육아", "어린이집", "아이돌봄", "출산", "양육", "돌봄교실", "영유아"),
    # 주의: "건강", "보건" 같은 단독어는 '건강보험 자격득실 확인서'(주거·복지 서류) 등에
    # 오탐되므로 쓰지 않는다. 실제 의료 도메인 어휘만 사용한다.
    "의료": ("의료", "병원", "의원", "진료", "건강검진", "보건소", "응급실", "간호", "치료", "약국"),
    "일자리": ("일자리", "취업", "구직", "채용", "창업", "직업", "훈련", "고용", "알선"),
    "교통": ("교통", "버스", "택시", "효도택시", "100원 택시", "맘편한 택시", "대중교통", "셔틀", "정류장", "이동권", "도로", "교통비", "이용권", "쿠폰"),
    "주거": ("주거", "주택", "임대", "빈집", "집수리", "정주", "청년주택", "거주"),
}

POLICY_ISSUE_TEMPLATES = {
    "공교육": [
        "학생·학부모 직접 수혜",
        "학교·지역 간 형평성",
        "면 지역 교육 접근성",
        "교육비 부담 완화",
        "프로그램 운영 인력과 지속성",
        "지역 학교 유지와 정주 효과",
    ],
    "보육": [
        "직접 수혜 대상",
        "읍면 접근성과 시설 편중",
        "운영 시간과 휴일 공백",
        "보육교사 인력과 처우",
        "아동 안전과 정서 안정",
        "예산 우선순위와 타 복지 균형",
        "정주 여건과 청년 가구 유입",
    ],
    "의료": [
        "고령자·환자 직접 수혜",
        "읍면 의료 접근성",
        "응급상황 대응 가능성",
        "의료 인력과 운영 지속성",
        "교통약자 이동 연계",
        "의료비·돌봄비 부담 완화",
    ],
    "일자리": [
        "구직자 직접 수혜",
        "직무훈련과 실제 취업 연결",
        "청년·중장년·경력단절 형평성",
        "읍면 접근성과 참여 장벽",
        "지역 기업 수요와 일자리 지속성",
        "생계 안정과 정책 우선순위",
    ],
    "교통": [
        "이용권 지급량·사용기한과 병원·장날 이동 수요",
        "기사 호출 수락·배차 실패와 참여 유인",
        "종이쿠폰·카드결제와 운행 정산 절차",
        "동승·공동배차의 편익과 이용 기준",
        "국비·도비·군비 분담과 운영 지속성",
        "교통약자·오지 주민의 집 앞 승하차와 직접 수혜",
        "이용 대상·지역 기준과 사각지대",
        "택시업계 운행 수요와 서비스 품질",
        "병원·시장·행정기관 접근성",
    ],
    "주거": [
        "청년·신혼부부 직접 수혜",
        "임대료·주거비 부담 완화",
        "입지와 생활 인프라",
        "빈집 활용과 마을 유지",
        "기존 주민과 신규 입주자 형평성",
        "교통·일자리와의 연결",
        "장기 정착 가능성",
    ],
    "일반": [
        "직접 수혜 대상",
        "읍면 접근성과 참여 장벽",
        "예산 우선순위와 형평성",
        "운영 인력과 지속성",
        "신청 절차와 정보 접근성",
        "취약계층 체감 효과",
    ],
}


@dataclass
class CitizenResponse:
    persona_id: str
    satisfaction_score: float
    summary_keyword: str
    opinion_text: str
    region_name_kr: str = ""
    age: int | None = None
    sex: str = ""
    occupation: str = ""
    stance: str = ""
    policy_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepresentativeAgent:
    persona_id: str
    satisfaction_score: float
    summary_keyword: str
    opinion_text: str
    cluster_id: str = ""
    cluster_name: str = ""
    selection_reason: str = ""
    region_name_kr: str = ""
    age: int | None = None
    age_group: str = ""
    sex: str = ""
    occupation: str = ""
    stance: str = ""
    conflict_risk: float = 0.0
    persona_policy_fit: float | None = None
    policy_id: str = ""
    resource_tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebatePair:
    pair_id: str
    asker_id: str
    target_id: str
    temperature_diff: float
    resource_overlap_bonus: float
    conflict_bonus: float
    pair_score: float
    shared_tags: list[str]
    reason: str


def read_text_auto(path: Path) -> str:
    for encoding in (UTF8_SIG, "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding=UTF8_SIG, errors="replace")


def load_json(path: Path) -> Any:
    return json.loads(read_text_auto(path))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding=UTF8_SIG,
    )


def load_env_files() -> None:
    for env_path in (SCRIPT_DIR / ".env", SCRIPT_DIR / "env"):
        if not env_path.exists():
            continue
        for raw_line in read_text_auto(env_path).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def clean_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def shorten(text: str, limit: int = 120) -> str:
    text = clean_space(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


_POLITE_ENDINGS = (
    (r"괜찮겠더라구요", "괜찮겠더라고요"),
    (r"편하겠구먼요?", "편할 것 같아요"), (r"좋겠구먼요?", "좋을 것 같아요"),
    (r"것 같아", "것 같아요"), (r"거 같아", "것 같아요"),
    (r"같아", "같아요"), (r"같네", "같아요"), (r"같다", "같아요"),
    (r"좋겠네", "좋겠어요"), (r"좋겠어", "좋겠어요"),
    (r"싶네", "싶어요"), (r"싶어", "싶어요"),
    (r"않네", "않아요"), (r"않아", "않아요"),
    (r"없네", "없어요"), (r"없어", "없어요"),
    (r"아쉽네", "아쉬워요"), (r"아쉬워", "아쉬워요"),
    (r"걱정이야", "걱정이에요"), (r"정책이야", "정책이에요"),
    (r"정책이네", "정책이에요"), (r"정책이지", "정책이에요"),
    (r"지원이지", "지원이에요"), (r"제도지", "제도예요"),
    (r"안심이야", "안심이에요"), (r"도움이야", "도움이 돼요"),
    (r"큰일이야", "큰일이에요"), (r"얘기야", "이야기예요"),
    (r"얘기네", "이야기네요"), (r"말이야", "말이에요"),
    (r"말이지", "말이에요"), (r"거야", "거예요"), (r"거다", "거예요"),
    (r"되겠지", "될 것 같아요"), (r"되겠어", "될 것 같아요"),
    (r"일이겠지", "일일 것 같아요"), (r"덜하겠지", "덜할 것 같아요"),
    (r"좋겠더라", "좋겠더라고요"), (r"없겠더라고", "없겠더라고요"),
    (r"고맙지", "고마워요"), (r"고마워", "고마워요"),
    (r"든든하네", "든든해요"), (r"든든해", "든든해요"),
    (r"반갑네", "반가워요"), (r"필요하네", "필요해요"),
    (r"생각해", "생각해요"), (r"느껴", "느껴요"),
    (r"알겠어", "알겠어요"), (r"편해졌어", "편해졌어요"),
    (r"바라봐", "바라고 있어요"), (r"바란다", "바랍니다"),
    (r"보인다", "보입니다"), (r"준다", "줍니다"),
    (r"된다네", "됩니다"), (r"된다", "됩니다"),
    (r"하다", "합니다"), (r"하네", "해요"),
    (r"급하거든", "급하거든요"), (r"힘들었거든", "힘들었거든요"),
    (r"들어", "들어요"), (r"봐", "봐요"), (r"몰라", "모르겠어요"),
    (r"해", "해요"),
)


def natural_polite(text: Any) -> str:
    """주민 발언의 의미를 유지하면서 반말·음슴체 종결을 자연스러운 해요체로 정리한다."""
    value = clean_space(text)
    if not value:
        return ""
    value = re.sub(r"(?<![가-힣])난\s+", "저는 ", value)
    value = re.sub(r"(?<![가-힣])나는\s+", "저는 ", value)
    value = re.sub(r"(?<![가-힣])내가\s+", "제가 ", value)
    value = value.replace(" 나한테", " 저에게").replace(" 내 생활", " 제 생활")
    value = value.replace(" 나 같은", " 저 같은").replace(" 내 입장", " 제 입장")
    value = value.replace(" 내겐", " 제게는").replace(" 나한텐", " 저에게는")
    value = value.replace("것 같아 아쉬운 마음", "것 같지만, 아쉬운 마음")
    # 한 문장 안에서 쉼표로 이어진 반말도 문맥에 맞는 연결형 존댓말로 정리한다.
    clause_endings = (
        (r"것 같아,", "것 같아서,"), (r"거 같아,", "것 같아서,"),
        (r"도움 안 되네,", "도움이 되지는 않지만,"), (r"필요 없네,", "필요하지는 않지만,"),
        (r"느껴져,", "느껴지지만,"), (r"보여,", "보이지만,"),
        (r"될 거야,", "될 것 같지만,"), (r"되는지 몰라,", "되는지 모르겠지만,"),
        (r"도움이 돼,", "도움이 돼서,"), (r"도움이 된다,", "도움이 됩니다."),
    )
    for source, target in clause_endings:
        value = re.sub(source, target, value)
    for source, target in _POLITE_ENDINGS:
        value = re.sub(rf"{source}(?=[.!?](?:\s|$)|$)", target, value)
    value = re.sub(r"될 거여(?=[.!?](?:\s|$)|$)", "될 거예요", value)
    value = re.sub(r"([가-힣]+)겠네(?=[.!?](?:\s|$)|$)", r"\1겠어요", value)
    value = re.sub(r"([가-힣]+) 있네(?=[.!?](?:\s|$)|$)", r"\1 있어요", value)
    value = re.sub(r"않을 듯(?=[.!?](?:\s|$)|$)", "않을 것 같아요", value)
    value = re.sub(r"좋겠고(?=[.!?](?:\s|$)|$)", "좋겠습니다", value)
    return clean_space(value)


# ─────────────────────────────────────────────────────────────────────────
# 렌더 직전 텍스트 후처리 클리너
# 페르소나/토론 내용은 절대 재생성하지 않는다. '문자열 조합 흔적'만 정리한다.
# ─────────────────────────────────────────────────────────────────────────
# doc_id / fact_id 패턴: IND-화순읍-007, PRAG-0609, PLAN-..., ORD-... 등 (괄호 포함 가능)
_DOC_ID_RE = re.compile(
    # 범위형(BUS-화순읍-001~005)과 다중 하이픈까지 통째로 제거하고, 감싼 괄호도 함께 정리
    r"\(?\b(?:IND|PLAN|ORD|NEWS|VK|MIN|PRM|BUS|PRAG|RAG|FACT)[-–][가-힣A-Za-z0-9]+(?:[-–~]\d+)*\b\)?"
)
# 발화에 새어나온 시스템/프롬프트 문구
_SYSTEM_PHRASES = (
    "배정 증거팩에 따르면", "배정된 증거팩에 따르면", "증거팩에 따르면",
    "배정 증거팩에", "배정된 증거팩", "배정 증거팩", "증거팩에 근거하면", "증거팩상",
)
# 입론 앞에 붙는 괄호 스탠스 라벨: "(대상 범위 확대 요구) ..."
_LEADING_STANCE_RE = re.compile(r"^\s*\(([^)]{2,25})\)\s*")


def _has_open_final(ch: str) -> bool:
    """한글 음절이 '받침 없음' 또는 'ㄹ받침'인지 — 조사 '(으)로' 판정용."""
    if not ("가" <= ch <= "힣"):
        return False
    jong = (ord(ch) - 0xAC00) % 28
    return jong == 0 or jong == 8  # 0: 받침 없음, 8: ㄹ


def strip_internal_identifiers(text: str, log: list[str] | None = None) -> str:
    """[A5] doc_id·시스템 문구·앞 괄호 스탠스 라벨을 제거한다(검출 시 로그)."""
    original = text
    text = _DOC_ID_RE.sub("", text)
    for phrase in _SYSTEM_PHRASES:
        text = text.replace(phrase, "")
    matched_stance = _LEADING_STANCE_RE.match(text)
    if matched_stance:
        text = _LEADING_STANCE_RE.sub("", text, count=1)
    text = re.sub(r"\(\s*\)", "", text)  # 식별자 제거로 빈 괄호가 남으면 제거
    text = clean_space(text)
    if log is not None and text != clean_space(original):
        log.append(f"[cleaner:A5 식별자제거] '{shorten(original, 46)}' → '{shorten(text, 46)}'")
    return text


def fix_double_endings(text: str) -> str:
    """[A1] '..', '한다.입니다.'처럼 중복된 종결을 하나로 축약."""
    text = re.sub(r"([.!?])[\s]*\1+", r"\1", text)  # ".." / ". ." → "."
    # '한다.입니다.' 류: 앞의 완결 종결(다/요) 뒤에 붙은 종결 조각을 제거
    text = re.sub(r"([다요])\.\s*(입니다|습니다|합니다|이다|됩니다|이었다|였다)\.?", r"\1.", text)
    return text


def fix_josa(text: str) -> str:
    """[A3] 종결 명사 뒤 조사 오류를 교정한다(특히 '(으)로')."""
    # 술어성 명사 + 으로 → '한 것으로'
    text = re.sub(r"(필요|불가피|시급)으로", r"\1한 것으로", text)

    def _ro(match: re.Match) -> str:
        ch = match.group(1)
        return ch + ("로" if _has_open_final(ch) else "으로")

    # 모음/ㄹ로 끝나는 음절 뒤의 '으로'는 '로'가 맞다(예: 기류으로→기류로, 지지으로→지지로)
    return re.sub(r"([가-힣])으로", _ro, text)


def fix_list_join(text: str) -> str:
    """[A4] 마침표로 끝나는 항목을 ', '로 이을 때 생기는 '.,'를 정리."""
    return re.sub(r"\.\s*,", ", ", text)


def clean_text(text: Any, log: list[str] | None = None) -> str:
    """렌더 직전 문자열 정리(A1·A3·A4·A5). 내용은 바꾸지 않는다."""
    if text is None:
        return ""
    result = str(text)
    result = strip_internal_identifiers(result, log)  # A5
    result = fix_list_join(result)                     # A4
    result = fix_double_endings(result)                # A1
    result = fix_josa(result)                          # A3
    return clean_space(result)


_CLEAN_SKIP_KEYS = {"id", "persona_id", "score", "overall_score", "stance", "keyword", "keywords"}


def deep_clean(value: Any, log: list[str] | None = None) -> Any:
    """report_data의 문자열 값을 재귀적으로 정리한다(식별자·조사·중복종결)."""
    if isinstance(value, str):
        return clean_text(value, log)
    if isinstance(value, list):
        return [deep_clean(item, log) for item in value]
    if isinstance(value, dict):
        return {
            key: (val if key in _CLEAN_SKIP_KEYS else deep_clean(val, log))
            for key, val in value.items()
        }
    return value


def _looks_like_sentence(text: str) -> bool:
    """짧은 요약 라벨(예: '고령층 지원 병행')과 온전한 문장을 구분한다(A2)."""
    text = clean_space(text)
    if not text:
        return False
    if re.search(r"(다|요|음|함|됨|까|죠)[.!?]?$", text):  # 종결어미로 끝나면 문장
        return True
    return len(text) >= 18  # 종결어미가 없어도 충분히 길면 문장으로 본다


def join_sentences(parts: list[str]) -> str:
    """[A1] 문장 조각을 이을 때 각 조각의 종결부호(?, !, .)를 보존하며 하나씩 마침한다.

    앞 조각이 의문문('…있습니까?')이면 물음표를 지우지 않는다. 종결부호가 없는 조각만
    마침표를 보태므로, 재질문(?)+보완요구(.)가 '…있습니까. …' 로 뭉개지지 않는다.
    """
    out: list[str] = []
    for part in parts:
        piece = clean_space(part).rstrip(" ，,")
        if not piece:
            continue
        if not re.search(r"[.!?…]$", piece):
            piece += "."
        out.append(piece)
    return " ".join(out)


def _age_text(age: int | None) -> str:
    return f"{age}세" if age is not None else ""


def _canonical_pid(persona_id: str) -> str:
    """발화 id에 붙은 접미사를 벗겨 정규 페르소나 ID를 얻는다.

    'HW017-OP1'→'HW017', 'HW085-1'→'HW085', 'rebuttal_HW041_to_HW072'→'HW041'.
    """
    pid = clean_space(persona_id)
    match = re.search(r"HW\d+", pid)
    if match:
        return match.group(0)
    return re.sub(r"[-–][A-Za-z0-9]+$", "", pid)


def normalize_speaker(agent: RepresentativeAgent | None, persona_id: str) -> dict[str, str]:
    """[A6/B4] 화자 라벨을 'HW085 · 81세' 한 포맷으로 통일하고, 상세는 따로 뗀다."""
    base = _canonical_pid(persona_id)  # 'HW085-1'/'HW017-OP1' → 'HW085'/'HW017'
    if agent is None:
        return {"name": base, "detail": ""}
    age = _age_text(agent.age)
    name = f"{base} · {age}" if age else base
    detail = " · ".join(part for part in (agent.region_name_kr, agent.occupation) if part)
    return {"name": name, "detail": detail}


def first_present(raw: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_response_record(raw: dict[str, Any]) -> CitizenResponse:
    known_keys = {
        "persona_id",
        "satisfaction_score",
        "summary_keyword",
        "opinion_text",
        "region_name_kr",
        "age",
        "sex",
        "occupation",
        "stance",
        "policy_id",
    }
    return CitizenResponse(
        persona_id=clean_space(first_present(raw, ("persona_id", "id", "시민ID"))),
        satisfaction_score=parse_float(first_present(raw, ("satisfaction_score", "만족도", "score")), 3.0),
        summary_keyword=clean_space(first_present(raw, ("summary_keyword", "키워드", "keyword"))),
        opinion_text=natural_polite(first_present(raw, ("opinion_text", "의견", "opinion"))),
        region_name_kr=clean_space(first_present(raw, ("region_name_kr", "지역", "region"))),
        age=parse_int(first_present(raw, ("age", "나이"))),
        sex=clean_space(first_present(raw, ("sex", "성별"))),
        occupation=clean_space(first_present(raw, ("occupation", "직업"))),
        stance=clean_space(first_present(raw, ("stance", "입장"))),
        policy_id=clean_space(first_present(raw, ("policy_id", "policy_name", "정책"))),
        extra={key: value for key, value in raw.items() if key not in known_keys},
    )


def load_policy_responses(path: Path) -> tuple[dict[str, Any], list[CitizenResponse], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"100명 의견 파일을 찾을 수 없습니다: {path}")

    if path.suffix.lower() == ".json":
        payload = load_json(path)
        if isinstance(payload, dict):
            policy = payload.get("policy") or {}
            rows = payload.get("results") or payload.get("responses") or payload.get("data") or []
            if isinstance(rows, dict):
                rows = list(rows.values())
            meta = {
                "summary": payload.get("summary") or {},
                "source_payload_type": "dict",
            }
        elif isinstance(payload, list):
            policy = {}
            rows = payload
            meta = {"summary": {}, "source_payload_type": "list"}
        else:
            raise ValueError(f"지원하지 않는 JSON 구조입니다: {path}")
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding=UTF8_SIG, newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        policy = {}
        meta = {"summary": {}, "source_payload_type": "csv"}
    else:
        raise ValueError(f"지원하지 않는 100명 의견 파일 형식입니다: {path.suffix}")

    responses = [
        normalize_response_record(row)
        for row in rows
        if isinstance(row, dict) and clean_space(first_present(row, ("persona_id", "id", "시민ID")))
    ]
    if not responses:
        raise ValueError(f"100명 의견 데이터를 읽었지만 유효한 persona_id가 없습니다: {path}")

    if not policy and responses[0].policy_id:
        policy = {
            "policy_name": responses[0].policy_id,
            "main_details": responses[0].policy_id,
        }

    return policy, responses, meta


def find_latest_representatives(output_root: Path) -> Path:
    candidates = list(output_root.glob("*/representative_agents.json"))
    direct = output_root / "representative_agents.json"
    if direct.exists():
        candidates.append(direct)
    if not candidates:
        raise FileNotFoundError(
            f"representative_agents.json을 찾을 수 없습니다. --cluster-dir 또는 --representatives를 지정하세요: {output_root}"
        )
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)[0]


def resolve_representative_path(args: argparse.Namespace) -> Path:
    if args.representatives:
        return Path(args.representatives).expanduser().resolve()
    if args.cluster_dir:
        return (Path(args.cluster_dir).expanduser().resolve() / "representative_agents.json")
    return find_latest_representatives(Path(args.output_root).expanduser().resolve())


def extract_resource_tags(*texts: str) -> list[str]:
    merged = " ".join(clean_space(text) for text in texts if text)
    tags: list[str] = []
    for tag, terms in RESOURCE_TAG_RULES.items():
        if any(term in merged for term in terms):
            tags.append(tag)
    return tags or ["기타"]


def infer_policy_category(policy_text: str) -> str:
    """가장 많이 일치하는 카테고리를 고른다.

    첫 일치를 바로 반환하면, 주거 정책이 '취업으로 독립하는 청년' 같은 부수 표현 때문에
    일자리로 오분류된다. 그래서 카테고리별 일치 횟수를 세어 최다 득표를 고른다.
    """
    text = clean_space(policy_text)
    best_category = "일반"
    best_score = 0
    for category, terms in POLICY_CATEGORY_RULES.items():
        score = sum(text.count(term) for term in terms)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def issue_tokens(issue: str) -> set[str]:
    text = clean_space(issue)
    tokens = {token for token in re.split(r"[^0-9A-Za-z가-힣]+", text) if len(token) >= 2}
    tags = set(extract_resource_tags(text))
    tags.discard("기타")
    return tokens | tags


def issues_overlap(candidate: str, selected: list[str]) -> bool:
    candidate_text = clean_space(candidate)
    candidate_tokens = issue_tokens(candidate_text)
    for issue in selected:
        issue_text = clean_space(issue)
        if candidate_text == issue_text or candidate_text in issue_text or issue_text in candidate_text:
            return True
        shared = candidate_tokens & issue_tokens(issue_text)
        if len(shared) >= 2:
            return True
    return False


def add_policy_tags_to_agents(agents: list[RepresentativeAgent], policy_text: str) -> None:
    policy_tags = [tag for tag in extract_resource_tags(policy_text) if tag != "기타"]
    if not policy_tags:
        return
    for agent in agents:
        existing = [tag for tag in agent.resource_tags if tag != "기타"]
        if existing:
            agent.resource_tags = existing
        else:
            agent.resource_tags = policy_tags or ["기타"]


def normalize_representative_record(
    raw: dict[str, Any],
    response_lookup: dict[str, CitizenResponse],
) -> RepresentativeAgent:
    persona_id = clean_space(first_present(raw, ("persona_id", "id", "시민ID")))
    source = response_lookup.get(persona_id)

    def raw_or_source(raw_keys: tuple[str, ...], source_attr: str, default: Any = "") -> Any:
        raw_value = first_present(raw, raw_keys, None)
        if raw_value not in (None, ""):
            return raw_value
        if source is not None:
            return getattr(source, source_attr)
        return default

    known_keys = {
        "persona_id",
        "satisfaction_score",
        "summary_keyword",
        "opinion_text",
        "cluster_id",
        "cluster_name",
        "selection_reason",
        "region_name_kr",
        "age",
        "age_group",
        "sex",
        "occupation",
        "stance",
        "conflict_risk",
        "persona_policy_fit",
        "policy_id",
    }

    agent = RepresentativeAgent(
        persona_id=persona_id,
        satisfaction_score=parse_float(raw_or_source(("satisfaction_score", "만족도", "score"), "satisfaction_score"), 3.0),
        summary_keyword=clean_space(raw_or_source(("summary_keyword", "키워드", "keyword"), "summary_keyword")),
        opinion_text=natural_polite(raw_or_source(("opinion_text", "의견", "opinion"), "opinion_text")),
        cluster_id=clean_space(first_present(raw, ("cluster_id", "군집ID"))),
        cluster_name=clean_space(first_present(raw, ("cluster_name", "군집명"))),
        selection_reason=clean_space(first_present(raw, ("selection_reason", "대표선정이유"))),
        region_name_kr=clean_space(raw_or_source(("region_name_kr", "지역", "region"), "region_name_kr")),
        age=parse_int(raw_or_source(("age", "나이"), "age", None)),
        age_group=clean_space(first_present(raw, ("age_group", "연령대"))),
        sex=clean_space(raw_or_source(("sex", "성별"), "sex")),
        occupation=clean_space(raw_or_source(("occupation", "직업"), "occupation")),
        stance=clean_space(raw_or_source(("stance", "입장"), "stance")),
        conflict_risk=parse_float(first_present(raw, ("conflict_risk", "갈등위험"), 0.0), 0.0),
        persona_policy_fit=parse_float(first_present(raw, ("persona_policy_fit", "정책적합도"), None), 0.0)
        if first_present(raw, ("persona_policy_fit", "정책적합도"), None) is not None
        else None,
        policy_id=clean_space(raw_or_source(("policy_id", "policy_name", "정책"), "policy_id")),
        extra={key: value for key, value in raw.items() if key not in known_keys},
    )
    agent.resource_tags = extract_resource_tags(
        agent.summary_keyword,
        agent.cluster_name,
        agent.opinion_text,
        agent.selection_reason,
    )
    return agent


def load_representatives(
    path: Path,
    responses: list[CitizenResponse],
) -> list[RepresentativeAgent]:
    if not path.exists():
        raise FileNotFoundError(f"대표 agent 파일을 찾을 수 없습니다: {path}")
    payload = load_json(path)
    if isinstance(payload, dict):
        rows = payload.get("representatives") or payload.get("agents") or payload.get("data") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"지원하지 않는 대표 agent JSON 구조입니다: {path}")
    lookup = {response.persona_id: response for response in responses}
    agents = [
        normalize_representative_record(row, lookup)
        for row in rows
        if isinstance(row, dict) and clean_space(first_present(row, ("persona_id", "id", "시민ID")))
    ]
    if not agents:
        raise ValueError(f"대표 agent가 비어 있습니다: {path}")
    return agents


def load_cluster_summary(cluster_dir: Path) -> list[dict[str, Any]]:
    path = cluster_dir / "cluster_summary.json"
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("clusters") or payload.get("data") or []
        return rows if isinstance(rows, list) else []
    return []


def load_cluster_run_summary(cluster_dir: Path) -> dict[str, Any]:
    path = cluster_dir / "run_summary.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def score_position(score: float, keyword: str = "") -> str:
    keyword = keyword or "정책"
    if score <= 1.5:
        return f"{keyword}에 대한 강한 우려"
    if score <= 2.5:
        return f"{keyword} 보완 필요"
    if score <= 3.5:
        return f"{keyword} 조건부 검토"
    if score <= 4.5:
        return f"{keyword} 조건부 기대"
    return f"{keyword} 강한 기대"


def persona_label(agent: RepresentativeAgent) -> str:
    parts = [agent.persona_id]
    details: list[str] = []
    if agent.region_name_kr:
        details.append(agent.region_name_kr)
    if agent.age is not None:
        details.append(f"{agent.age}세")
    if agent.occupation:
        details.append(agent.occupation)
    if details:
        parts.append(f"({', '.join(details)})")
    return " ".join(parts)


# 정책-무관 '방관자 화법' 감지.
# 정책마다 달라지는 수혜 어휘(장학금·버스·진료…)를 나열하는 대신,
# 도메인과 무관한 '직접 도움은 없다'류 거리두기 표현으로 간접수혜(비수혜) 대표를 판별한다.
# 이렇게 하면 공교육뿐 아니라 교통·보건·농업 등 어떤 정책에도 그대로 재사용된다.
_BYSTANDER_DIRECT_NEG = re.compile(r"직접[가-힣\s]{0,18}(아니|없|못|않|미미|부족|어렵|안 되|안 돼|안 느)")
BYSTANDER_MARKERS = (
    "와닿지", "와 닿지", "와닿진", "와 닿진", "와닿는 건 아니", "와 닿는 건 아니",
    "와닿는 게 별로", "와 닿는 게 별로", "해당 없", "해당사항 없",
    "상관없", "상관 없", "관계없", "관계 없",
    "체감 안", "체감이 안", "체감하기 어", "체감은 적", "체감이 낮",
    "느끼진 않", "느끼지 못", "멀게 느", "거리감", "거리가 있", "거리가 좀",
    "남 일", "겪는 일은 아니", "겪을 일", "내 일이 아니", "일은 아니",
    "생활에 크게", "생활에 큰", "생활에서 크게", "삶에 크게", "삶에 큰", "일상에 크게", "일상엔",
    "변화 없", "큰 변화", "달라지는 건 없", "달라지는 게 없", "달라질 건 없",
    "건 아니지만", "건 아니고", "것도 아니", "것은 아니",
)


def benefit_stance(agent: RepresentativeAgent) -> str:
    """대표 시민이 정책의 직접 수혜자인지(직접수혜) 방관자인지(간접수혜)를 화법으로 추정한다.

    특정 정책 어휘가 아니라 도메인-무관한 거리두기 표현으로 판정하므로 정책이 바뀌어도 재사용된다.
    (가족 명사 기반 '손주 없다'류는 일부러 잡지 않는다 — 그래야 교육에 다시 묶이지 않는다.)
    """
    text = f"{agent.opinion_text} {agent.summary_keyword}"
    if _BYSTANDER_DIRECT_NEG.search(text) or any(marker in text for marker in BYSTANDER_MARKERS):
        return "간접수혜"
    return "직접수혜"


def response_benefit_stance(response: CitizenResponse) -> str:
    """CitizenResponse에 대해서도 같은 '직접/간접수혜' 화법 판정을 적용한다."""
    text = f"{response.opinion_text} {response.summary_keyword}"
    if _BYSTANDER_DIRECT_NEG.search(text) or any(marker in text for marker in BYSTANDER_MARKERS):
        return "간접수혜"
    return "직접수혜"


# 대표 중복 판정에서 잡음이 되는 2자 이상 기능어(1자어는 아래 토큰화에서 이미 제외됨).
_REP_STOPWORDS = {
    "그래도", "그리고", "하지만", "이런", "저런", "정도", "관련", "그런", "우리",
    "나는", "나도", "내가", "정말", "그냥", "많이", "때문",
}


def _opinion_tokens(agent: RepresentativeAgent) -> set[str]:
    text = f"{agent.opinion_text} {agent.summary_keyword}"
    tokens = {
        token
        for token in re.split(r"[^0-9A-Za-z가-힣]+", text)
        if len(token) >= 2 and token not in _REP_STOPWORDS
    }
    tokens |= {tag for tag in agent.resource_tags if tag != "기타"}
    return tokens


def representatives_similar(a: RepresentativeAgent, b: RepresentativeAgent, jaccard: float = 0.22, shared: int = 3) -> bool:
    """두 대표가 '사실상 같은 주장'인지 내용 중심으로 판정한다.

    페르소나의 나이·지역이 달라도 동일 입장과 동일 핵심어를 반복하면 중복으로 본다.
    반대로 같은 쟁점을 두고 찬반 또는 직접·간접 수혜 입장이 다르면 토론 가치가 있으므로 남긴다.
    """
    if spec_stance(a.satisfaction_score, a.stance) != spec_stance(b.satisfaction_score, b.stance):
        return False
    if benefit_stance(a) != benefit_stance(b):
        return False
    keyword_a = clean_space(a.summary_keyword).replace(" ", "")
    keyword_b = clean_space(b.summary_keyword).replace(" ", "")
    if keyword_a and keyword_b and keyword_a == keyword_b:
        return True
    tokens_a, tokens_b = _opinion_tokens(a), _opinion_tokens(b)
    if not tokens_a or not tokens_b:
        return False
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return (intersection / union) >= jaccard or intersection >= shared


def dedupe_representatives(
    representatives: list[RepresentativeAgent],
    warnings: list[str],
) -> list[RepresentativeAgent]:
    """서로 같은 주장을 반복하는 대표를 통합해, 대표끼리는 다른 목소리만 남긴다.

    (군집화가 비슷한 의견을 다른 키워드·페르소나 때문에 다른 군집으로 쪼개 대표가
    중복되는 경우를 downstream에서 정리한다. 먼저 등장한 대표를 유지한다.)
    """
    kept: list[RepresentativeAgent] = []
    for agent in representatives:
        duplicate = next((other for other in kept if representatives_similar(agent, other)), None)
        if duplicate is not None:
            warnings.append(
                f"유사 주장 대표 통합: {agent.persona_id}({agent.summary_keyword}) ≈ "
                f"{duplicate.persona_id}({duplicate.summary_keyword}) — 중복 제거"
            )
            continue
        kept.append(agent)
    return kept


def response_to_representative(response: CitizenResponse, selection_reason: str) -> RepresentativeAgent:
    """100명 응답 중 한 명을 토론 대표로 승격시킨다(직접 수혜·찬성 대표 보강용)."""
    agent = RepresentativeAgent(
        persona_id=response.persona_id,
        satisfaction_score=response.satisfaction_score,
        summary_keyword=response.summary_keyword,
        opinion_text=response.opinion_text,
        cluster_name="직접수혜·찬성",
        selection_reason=selection_reason,
        region_name_kr=response.region_name_kr,
        age=response.age,
        sex=response.sex,
        occupation=response.occupation,
        stance=response.stance,
        policy_id=response.policy_id,
    )
    agent.resource_tags = extract_resource_tags(agent.summary_keyword, agent.opinion_text)
    return agent


def ensure_direct_beneficiary_representative(
    representatives: list[RepresentativeAgent],
    responses: list[CitizenResponse],
    warnings: list[str],
    min_score: int = 4,
) -> list[RepresentativeAgent]:
    """토론 대표에 '직접 수혜자이면서 찬성(만족도 높음)'인 시민이 최소 1명 있도록 보강한다.

    군집화가 소수 입장(직접 수혜 찬성층)을 대표로 못 뽑으면, 당사자 목소리가 토론에서
    빠져 개선 요구가 '비수혜층 형평' 쪽으로 쏠린다. 그래서 여기서 한 명을 강제로 넣는다.
    후보가 여럿이면 만족도가 가장 높은 층에서 무작위로 한 명을 뽑는다.
    """
    for agent in representatives:
        if benefit_stance(agent) == "직접수혜" and agent.satisfaction_score >= min_score:
            return representatives  # 이미 직접 수혜·찬성 대표가 있음

    existing_ids = {agent.persona_id for agent in representatives}
    candidates = [
        response
        for response in responses
        if response.persona_id not in existing_ids
        and response.satisfaction_score >= min_score
        and response_benefit_stance(response) == "직접수혜"
    ]
    if not candidates:
        warnings.append(
            "직접 수혜·찬성 대표 보강 실패: 직접 수혜자이면서 찬성(만족도 높음)인 시민을 찾지 못함."
        )
        return representatives

    top_score = max(response.satisfaction_score for response in candidates)
    top_candidates = [r for r in candidates if r.satisfaction_score == top_score]
    chosen = random.choice(top_candidates)  # 여러 명이면 무작위 선발
    new_agent = response_to_representative(
        chosen, f"직접 수혜·찬성 대표 자동 보강(만족도 {chosen.satisfaction_score:g}점, 후보 {len(top_candidates)}명 중 선발)"
    )
    warnings.append(
        f"직접 수혜·찬성 대표 {chosen.persona_id}({chosen.summary_keyword})을(를) 토론에 보강함."
    )
    return [*representatives, new_agent]


def is_myeon(agent: RepresentativeAgent) -> bool:
    return agent.region_name_kr.endswith("면")


def interest_contrast(agent_a: RepresentativeAgent, agent_b: RepresentativeAgent) -> tuple[float, list[str]]:
    """만족도 온도차 대신, 이해관계가 얼마나 다른지를 점수와 축 이름으로 돌려준다.

    같은 쟁점(예: 교육)을 두고 서로 처지가 다른 두 대표를 붙여야
    '보완점'이 드러나는 반대신문이 된다. 합성 데이터에 만족도 온도차가
    거의 없으므로, 직접 수혜 여부·읍면 접근성·세대 차이를 1순위로 본다.
    """
    axes: list[str] = []
    bonus = 0.0
    if benefit_stance(agent_a) != benefit_stance(agent_b):
        bonus += 2.5
        axes.append("직접수혜 여부")
    if is_myeon(agent_a) != is_myeon(agent_b):
        bonus += 1.5
        axes.append("읍/면 접근성")
    if (
        agent_a.age is not None
        and agent_b.age is not None
        and abs(agent_a.age - agent_b.age) >= 20
    ):
        bonus += 1.0
        axes.append("세대 차이")
    return round(bonus, 3), axes


def select_debate_pairs(
    agents: list[RepresentativeAgent],
    max_pairs: int = 3,
) -> list[DebatePair]:
    candidates: list[
        tuple[float, float, RepresentativeAgent, RepresentativeAgent, list[str], list[str], float, float, float]
    ] = []
    for index, agent_a in enumerate(agents):
        for agent_b in agents[index + 1 :]:
            temperature_diff = abs(agent_a.satisfaction_score - agent_b.satisfaction_score)
            shared_tags = sorted(set(agent_a.resource_tags) & set(agent_b.resource_tags))
            shared_tags = [tag for tag in shared_tags if tag != "기타"]
            resource_overlap_bonus = 2.0 if shared_tags else 0.0
            conflict_bonus = round((agent_a.conflict_risk + agent_b.conflict_risk) / 2, 3)
            contrast_bonus, contrast_axes = interest_contrast(agent_a, agent_b)
            # 이해관계 대비를 1순위로, 공통 쟁점(같은 싸움터)과 만족도 온도차는 보조 지표로만.
            pair_score = round(
                contrast_bonus + resource_overlap_bonus + conflict_bonus + temperature_diff,
                3,
            )
            candidates.append(
                (
                    pair_score,
                    contrast_bonus,
                    agent_a,
                    agent_b,
                    shared_tags,
                    contrast_axes,
                    temperature_diff,
                    resource_overlap_bonus,
                    conflict_bonus,
                )
            )

    candidates.sort(key=lambda item: (item[0], item[1], item[6], item[7]), reverse=True)
    selected: list[DebatePair] = []
    selected_keys: set[frozenset[str]] = set()
    appearances: Counter[str] = Counter()
    # 커버리지 목표: 모든 대표가 최소 1회 반대신문에 참여하도록(=방관자 없음),
    # n명을 덮는 데 필요한 최소 쌍 수 ceil(n/2)까지 확보한다.
    target_pairs = min(max_pairs, max(1, math.ceil(len(agents) / 2)))

    def append_pair(record: tuple[Any, ...]) -> None:
        (
            pair_score,
            _contrast_bonus,
            agent_a,
            agent_b,
            shared_tags,
            contrast_axes,
            temperature_diff,
            resource_overlap_bonus,
            conflict_bonus,
        ) = record
        stance_a = benefit_stance(agent_a)
        stance_b = benefit_stance(agent_b)
        if stance_a != stance_b:
            # 직접 수혜가 적은(간접수혜) 쪽이 수혜자에게 체감 격차를 묻는 구도를 우선한다.
            asker, target = (agent_a, agent_b) if stance_a == "간접수혜" else (agent_b, agent_a)
        elif agent_a.satisfaction_score < agent_b.satisfaction_score:
            asker, target = agent_a, agent_b
        elif agent_b.satisfaction_score < agent_a.satisfaction_score:
            asker, target = agent_b, agent_a
        elif agent_a.conflict_risk >= agent_b.conflict_risk:
            asker, target = agent_a, agent_b
        else:
            asker, target = agent_b, agent_a

        reason_parts: list[str] = []
        if contrast_axes:
            reason_parts.append("이해관계 대비 " + ", ".join(contrast_axes))
        if shared_tags:
            reason_parts.append(f"공통 쟁점 {', '.join(shared_tags)}")
        if temperature_diff:
            reason_parts.append(f"만족도 온도차 {temperature_diff:g}점")
        reason = " / ".join(reason_parts) or "이해관계 대비가 뚜렷하지 않아 기본 쌍으로 선정"

        selected.append(
            DebatePair(
                pair_id=f"P{len(selected) + 1}",
                asker_id=asker.persona_id,
                target_id=target.persona_id,
                temperature_diff=round(temperature_diff, 3),
                resource_overlap_bonus=resource_overlap_bonus,
                conflict_bonus=conflict_bonus,
                pair_score=pair_score,
                shared_tags=shared_tags,
                reason=reason,
            )
        )
        selected_keys.add(frozenset((agent_a.persona_id, agent_b.persona_id)))
        appearances[agent_a.persona_id] += 1
        appearances[agent_b.persona_id] += 1

    # 1차: 서로 겹치지 않는 고득점 쌍부터(자연스러운 최고 대비 매칭).
    for record in candidates:
        if len(selected) >= target_pairs:
            break
        agent_a, agent_b = record[2], record[3]
        if appearances[agent_a.persona_id] == 0 and appearances[agent_b.persona_id] == 0:
            append_pair(record)

    # 2차: 아직 한 번도 참여 못한 대표를, 대비가 큰 상대와 붙여 끌어들인다(상대는 최대 2회까지 재등장 허용).
    while len(selected) < max_pairs:
        uncovered = {agent.persona_id for agent in agents if appearances[agent.persona_id] == 0}
        if not uncovered:
            break
        pick = None
        for record in candidates:
            agent_a, agent_b = record[2], record[3]
            key = frozenset((agent_a.persona_id, agent_b.persona_id))
            if key in selected_keys:
                continue
            if agent_a.persona_id not in uncovered and agent_b.persona_id not in uncovered:
                continue
            if appearances[agent_a.persona_id] >= 2 or appearances[agent_b.persona_id] >= 2:
                continue
            pick = record
            break
        if pick is None:
            break
        append_pair(pick)

    return selected


def infer_evidence_issue(agent: RepresentativeAgent, policy_text: str) -> str:
    # 정책 원문에서 추출한 자원 쟁점과 대표의 쟁점이 겹치면 그것을 최우선으로 사용한다.
    # (특정 도메인을 하드코딩하지 않아 어떤 정책에도 동작한다.)
    agent_tags = [tag for tag in agent.resource_tags if tag != "기타"]
    policy_tags = [tag for tag in extract_resource_tags(policy_text) if tag != "기타"]
    for tag in policy_tags:
        if tag in agent_tags:
            return tag
    if policy_tags:
        return policy_tags[0]
    if agent_tags:
        return agent_tags[0]
    return agent.summary_keyword or "__no_issue__"


def build_evidence_packs(
    agents: list[RepresentativeAgent],
    policy_text: str,
    db_path: Path,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    packs: dict[str, dict[str, Any]] = {}
    if not db_path.exists():
        warnings.append(f"증거팩 DB를 찾지 못해 fact_id 검증은 빈 팩으로 진행합니다: {db_path}")
        return {agent.persona_id: {"cards": [], "사용_규칙": []} for agent in agents}
    try:
        from evidence_pack_builder import build_pack
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"증거팩 빌더 import 실패: {type(exc).__name__}: {exc}")
        return {agent.persona_id: {"cards": [], "사용_규칙": []} for agent in agents}

    for agent in agents:
        issue = infer_evidence_issue(agent, policy_text)
        tags = [tag for tag in agent.resource_tags if tag != "기타"]
        try:
            # Before-stage leakage control lives in evidence_pack_builder.policy_plan_cards:
            # policy_plan cards are restricted to rag_stage=simulation_context and leakage_risk=low.
            packs[agent.persona_id] = build_pack(str(db_path), agent.region_name_kr or "화순읍", issue, tags)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"증거팩 생성 실패({agent.persona_id}, {issue}): {type(exc).__name__}: {exc}")
            packs[agent.persona_id] = {"cards": [], "사용_규칙": []}
    return packs


def evidence_cards(pack: dict[str, Any] | None) -> list[dict[str, Any]]:
    cards = (pack or {}).get("cards") or []
    return [card for card in cards if isinstance(card, dict)]


def evidence_fact_ids(pack: dict[str, Any] | None) -> set[str]:
    return {str(card.get("fact_id")) for card in evidence_cards(pack) if card.get("fact_id")}


def first_fact_id(pack: dict[str, Any] | None) -> str:
    cards = evidence_cards(pack)
    preferred_order = ("PLAN", "ORD", "NEWS", "IND", "VK", "MIN", "PRM", "BUS")
    for prefix in preferred_order:
        for card in cards:
            fact_id = str(card.get("fact_id") or "")
            if fact_id.startswith(prefix):
                return fact_id
    ids = sorted(evidence_fact_ids(pack))
    return ids[0] if ids else ""


def format_evidence_pack(pack: dict[str, Any] | None, limit: int = 8) -> str:
    lines: list[str] = []
    for card in evidence_cards(pack)[:limit]:
        lines.append(
            f"- fact_id={card.get('fact_id')} | 종류={card.get('종류')} | 내용={shorten(card.get('내용', ''), 180)}"
        )
    return "\n".join(lines) if lines else f"- {UNVERIFIED_TEXT}"


def cited_fact_ids(statement: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    ground = statement.get("ground")
    if isinstance(ground, dict):
        value = ground.get("인용_fact_id")
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
        elif value:
            ids.append(str(value))
    for key in ("인용_fact_id", "근거_fact_id"):
        value = statement.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
        elif value:
            ids.append(str(value))
    return list(dict.fromkeys(ids))


def statement_text(statement: dict[str, Any]) -> str:
    chunks: list[str] = []
    for value in statement.values():
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, dict):
            chunks.extend(str(item) for item in value.values() if isinstance(item, (str, int, float)))
        elif isinstance(value, list):
            chunks.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    return " ".join(chunks)


def visible_opening_text(opening: dict[str, Any]) -> str:
    """보고서 화면에 실제 표시되는 입론만 반환한다.

    반대신문은 대표자의 숨은 원본 의견이나 프롬프트 정보가 아니라 이 문자열만
    대화 기억으로 사용해야 한다.
    """
    position = clean_space(opening.get("입장") or opening.get("claim"))
    ground = clean_space(opening.get("핵심근거") or (opening.get("ground") or {}).get("내용"))
    position_text = format_position_statement(position)
    return join_sentences([position_text, ground])


def format_position_statement(position: str) -> str:
    """입장 문장에 '저는'이 이미 있으면 중복으로 감싸지 않는다."""
    position = natural_polite(clean_space(position))
    if not position:
        return ""
    if position.startswith(("저는 ", "제 입장은 ")):
        return position
    return f"저는 이 정책에 대해 '{position}'라는 입장입니다."


def stance_polarity(text: str, score: float | None = None) -> str:
    """조건부 표현을 유지한 채 입장의 찬성·반대·중립 방향만 판별한다."""
    value = clean_space(text)
    if "반대" in value:
        return "반대"
    if "찬성" in value:
        return "찬성"
    if any(term in value for term in ("중립", "유보", "판단을 보류")):
        return "중립"
    if score is not None:
        if score >= 3.5:
            return "찬성"
        if score <= 2.5:
            return "반대"
    return "중립"


def ground_question_in_visible_opening(
    question: dict[str, Any],
    target_opening: dict[str, Any],
) -> dict[str, Any]:
    """AI 질문의 인용부를 화면에 나온 실제 입론 문장으로 강제한다."""
    grounded = dict(question)
    quote = clean_space(
        target_opening.get("핵심근거")
        or (target_opening.get("ground") or {}).get("내용")
        or target_opening.get("입장")
        or target_opening.get("claim")
    )
    if not quote:
        return grounded
    original = clean_space(grounded.get("질문"))
    # AI가 만든 잘못된 '앞서 ... 말씀하셨는데' 부분은 버리고 질문의 본론만 살린다.
    tail = re.sub(r"^앞서\s+.+?라고\s+말씀하셨는데,?\s*", "", original)
    if not tail or tail == original:
        tail = "이 입장을 실제 운영에 적용하려면 어떤 보완 조건이 필요하다고 생각하십니까?"
    grounded["상대발언인용"] = quote
    grounded["질문"] = f"앞서 '{quote}'라고 말씀하셨는데, {tail}"
    return grounded


def extract_numbers(text: str) -> list[str]:
    cleaned = text or ""
    cleaned = re.sub(r"HW\d+", "", cleaned)
    cleaned = re.sub(r"[A-Z]+-[^\s,|]+-\d+", "", cleaned)
    cleaned = re.sub(r"\d+(?:\.\d+)?\s*(?:세|살)", "", cleaned)
    return re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", cleaned)


_GROUNDING_STOPWORDS = {
    "저는", "정책", "생각", "말씀", "부분", "경우", "사람", "시민", "화순군",
    "필요", "대한", "있습니다", "합니다", "같습니다", "그렇지만", "그리고",
}


def grounding_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[가-힣]{2,}", clean_space(text))
        if token not in _GROUNDING_STOPWORDS and len(token) >= 2
    }


def validate_statement(
    statement: dict[str, Any],
    *,
    stage: str,
    pack: dict[str, Any] | None = None,
    target_quote_source: str = "",
    expected_stance: str = "",
) -> dict[str, Any]:
    failures: list[str] = []
    fact_ids = evidence_fact_ids(pack)
    cited = cited_fact_ids(statement)
    text = statement_text(statement)

    if stage in {"opening", "rebuttal"}:
        if not cited:
            failures.append("missing_fact_id")
        else:
            invalid = [fact_id for fact_id in cited if fact_id not in fact_ids]
            if invalid:
                failures.append(f"invalid_fact_id:{','.join(invalid)}")

    if stage == "opening" and expected_stance:
        expected = stance_polarity(expected_stance)
        actual = stance_polarity(clean_space(statement.get("입장") or statement.get("claim")))
        if expected != actual:
            failures.append(f"stance_mismatch:{expected}!={actual}")

    if stage == "rebuttal":
        quote = clean_space(statement.get("상대발언인용") or statement.get("지목_문장") or "")
        source = clean_space(target_quote_source)
        if not quote or (source and quote not in source):
            failures.append("missing_or_unmatched_target_quote")
        question = clean_space(statement.get("질문"))
        quote_terms = grounding_terms(quote)
        if quote_terms and not (quote_terms & grounding_terms(question)):
            failures.append("question_not_linked_to_quoted_opening")

    if stage == "answer" and statement.get("답변유형") not in ANSWER_TYPE_ENUM:
        failures.append("invalid_answer_type")
    if stage == "answer" and target_quote_source:
        source_terms = grounding_terms(target_quote_source)
        answer_terms = grounding_terms(clean_space(statement.get("답변")))
        if source_terms and not (source_terms & answer_terms):
            failures.append("answer_not_linked_to_original_opening")

    cited_card_text = " ".join(
        str(card.get("내용", ""))
        for card in evidence_cards(pack)
        if str(card.get("fact_id")) in set(cited)
    )
    numbers = extract_numbers(text)
    if numbers and cited:
        missing_numbers = [number for number in numbers if number not in cited_card_text]
        if missing_numbers:
            failures.append(f"number_mismatch:{','.join(missing_numbers)}")
    elif numbers and not cited and stage in {"opening", "rebuttal"}:
        failures.append("number_without_fact_id")

    return {
        "stage": stage,
        "passed": not failures,
        "failures": failures,
        "cited_fact_ids": cited,
        "numbers": numbers,
        "number_checked": bool(numbers),
    }


def attach_verification(statement: dict[str, Any], result: dict[str, Any], regenerated: bool = False) -> dict[str, Any]:
    statement["_verification"] = result
    statement["_verification_failed"] = not result["passed"]
    statement["_regenerated"] = regenerated
    return statement


def validation_metrics(statements: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [item.get("_verification") for item in statements if isinstance(item.get("_verification"), dict)]
    checked = len(checks)
    cited_checks = [check for check in checks if check.get("stage") in {"opening", "rebuttal"}]
    numeric_checks = [check for check in checks if check.get("number_checked")]
    valid_citations = [
        check for check in cited_checks
        if check.get("cited_fact_ids") and not any(str(f).startswith("invalid_fact_id") for f in check.get("failures", []))
    ]
    numeric_passes = [
        check for check in numeric_checks
        if not any(str(f).startswith("number_mismatch") or f == "number_without_fact_id" for f in check.get("failures", []))
    ]
    return {
        "checked_statements": checked,
        "verification_failure_count": sum(1 for check in checks if not check.get("passed")),
        "valid_fact_citation_rate": round(len(valid_citations) / len(cited_checks), 4) if cited_checks else None,
        "numeric_alignment_rate": round(len(numeric_passes) / len(numeric_checks), 4) if numeric_checks else None,
        "numeric_alignment_note": "수치 발언이 있는 경우에만 계산; 없으면 null",
    }


def interest_profile(agent: RepresentativeAgent) -> str:
    if agent.satisfaction_score >= 4:
        return "정책 기대가 있지만 전달 실패, 지속성 부족, 실제 수혜 누락이 있으면 잃을 것이 있는 입장"
    if agent.satisfaction_score <= 2:
        return "현재 조건에서는 체감 이익보다 소외나 비용 부담을 더 크게 볼 수 있는 입장"
    return "정책 취지는 이해하지만 직접 체감, 공정성, 실행 조건이 불명확하면 판단을 보류하는 입장"


class LocalDebateClient:
    def opening(self, agent: RepresentativeAgent, policy_text: str, evidence_pack: dict[str, Any] | None = None) -> dict[str, Any]:
        fact_id = first_fact_id(evidence_pack)
        # API 검증 실패 시에도 기존 찬반 입장을 바꾸지 않는 안전한 대체 입론.
        claim = clean_space(agent.stance) or score_position(agent.satisfaction_score, agent.summary_keyword)
        reason = natural_polite(shorten(agent.opinion_text, 150))
        return {
            "id": agent.persona_id,
            "발언유형": "입론",
            "입장": claim,
            "만족도": int(round(agent.satisfaction_score)),
            "claim": claim,
            "ground": {
                "내용": reason,
                "인용_fact_id": fact_id,
            },
            "warrant": "대표자의 기존 의견과 배정된 증거팩 범위에서 체감 조건을 설명합니다.",
            "핵심근거": reason,
        }

    def question(
        self,
        asker: RepresentativeAgent,
        target: RepresentativeAgent,
        pair: DebatePair,
        policy_text: str,
        target_opening: dict[str, Any] | None = None,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        shared = ", ".join(pair.shared_tags) if pair.shared_tags else f"{asker.summary_keyword}와 {target.summary_keyword}"
        fact_id = first_fact_id(evidence_pack)
        target_quote = clean_space(
            (target_opening or {}).get("핵심근거")
            or (target_opening or {}).get("claim")
            or target.opinion_text
        )
        return {
            "from_id": asker.persona_id,
            "to_id": target.persona_id,
            "발언유형": "반박",
            "지목_대상": target.persona_id,
            "상대발언인용": target_quote,
            "쟁점": shared,
            "claim": f"{asker.summary_keyword or shared} 관점에서는 {target.persona_id}의 조건을 그대로 받아들이기 어렵습니다.",
            "ground": {
                "내용": shorten(asker.opinion_text, 120),
                "인용_fact_id": fact_id,
            },
            "warrant": "내 이해관계에서 잃을 수 있는 체감 조건을 기준으로 상대 발언의 적용 한계를 묻습니다.",
            # 상대 발언을 그대로 인용하지 않고 상대의 핵심 관점(키워드)으로 요약해 지목한다.
            "질문": (
                f"앞서 '{shorten(target_quote, 38)}'라고 말씀하셨는데, "
                f"{asker.region_name_kr or '다른 지역'}의 {asker.age or ''}세 시민처럼 "
                f"{asker.summary_keyword} 문제를 느끼는 사람에게는 어떤 보완 조건이 있어야 "
                "정책 효과가 실제로 체감된다고 생각하십니까?"
            ),
        }

    def answer(
        self,
        target: RepresentativeAgent,
        asker: RepresentativeAgent,
        question: dict[str, Any],
        policy_text: str,
        target_opening: dict[str, Any] | None = None,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target_tags = set(target.resource_tags)
        asker_tags = set(asker.resource_tags)
        shared = sorted((target_tags | asker_tags) - {"기타"})
        issue = shared[0] if shared else asker.summary_keyword or "보완 조건"
        if target.satisfaction_score >= 4:
            answer_type = "조건부수용"
            answer_text = f"{issue} 보완이 빠지면 기대가 줄어드는 점은 인정하며, 현장 접근 방식이나 단계적 지원 조건을 함께 넣어야 합니다."
        elif target.satisfaction_score <= 2:
            answer_type = "입장고수"
            answer_text = f"{issue} 문제가 해결되지 않으면 이 정책이 내 생활에 닿기 어렵다는 판단은 유지합니다."
        else:
            answer_type = "부분수용"
            answer_text = f"{issue} 조건이 명확해지면 판단을 조금 더 긍정적으로 바꿀 수 있습니다."
        original_position = clean_space(
            (target_opening or {}).get("입장") or (target_opening or {}).get("claim")
        ) or score_position(target.satisfaction_score, target.summary_keyword)
        answer_text = f"제 입장은 여전히 {original_position}입니다. {answer_text}"
        return {
            "id": target.persona_id,
            "발언유형": "답변",
            "지목_대상": asker.persona_id,
            "답변유형": answer_type,
            "답변": natural_polite(answer_text),
        }

    def followup(
        self,
        asker: RepresentativeAgent,
        target: RepresentativeAgent,
        question: dict[str, Any],
        answer: dict[str, Any],
        policy_text: str,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 재반박 라운드: 답변을 듣고 '실제로 해소됐는지 + 남은 보완 요구'를 1턴 더 짚는다.
        resolved = clean_space(answer.get("답변유형")) in {"조건부수용", "부분수용"}
        accept = "일부해소" if resolved else "미해소"
        issue = asker.summary_keyword or "체감 조건"
        return {
            "id": asker.persona_id,
            "발언유형": "재질문",
            "지목_대상": target.persona_id,
            "수용여부": accept,
            "재질문": (
                f"방금 답변만으로 {issue} 문제가 실제로 해소되는지, "
                "누가 어떤 경우에 빠지는지 구체적 기준을 다시 여쭙니다."
            ),
            # 답변을 그대로 인용하지 않고 '방금 답변만으로는 부족하다'로 요약해 짚는다.
            "보완요구": (
                f"{issue} 관점에서 방금 답변만으로는 부족하니, "
                "소외되는 대상과 적용 조건을 구체적으로 정해 정책에 넣어야 합니다."
            ),
        }

    def issue_mediation(
        self,
        issue: str,
        agents: list[RepresentativeAgent],
        policy_text: str,
        answers: list[dict[str, Any]],
        evidence_packs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sorted_agents = sorted(agents, key=lambda item: item.satisfaction_score)
        low = sorted_agents[0] if sorted_agents else None
        high = sorted_agents[-1] if sorted_agents else None
        # 진행자는 '수용 조건'을 묻기보다 쟁점 상황을 제시하고 의견을 청한다.
        moderator_question = f"{issue} 부분에서 대표들 간 입장 차이가 있습니다. 이에 대해 어떻게 생각하십니까?"
        responses: list[dict[str, str]] = []
        for agent in sorted_agents[:3]:
            personal_issue = agent.summary_keyword or "제 생활에서 느끼는 불편"
            if agent.satisfaction_score <= 2:
                opinion = (
                    f"저는 {personal_issue} 문제 때문에 아직 정책 효과를 크게 체감하기 어렵습니다. "
                    f"{issue} 문제를 논의할 때 직접 혜택을 받지 못하는 주민의 사정도 함께 봐야 한다고 생각합니다."
                )
                concession = f"{issue}에서 빠지는 대상과 보완 절차가 분명해진다면 일부 수용할 수 있습니다."
            elif agent.satisfaction_score >= 4:
                opinion = (
                    f"저는 {personal_issue} 측면에서 정책에 기대하는 부분이 있습니다. "
                    f"다만 {issue}까지 실제 운영에 반영돼야 현장에서 도움이 될 것 같습니다."
                )
                concession = f"전면 시행이 어렵다면 {issue}과 관련해 필요가 큰 지역과 시간대부터 시작해도 괜찮습니다."
            else:
                opinion = (
                    f"저는 {personal_issue}에는 공감하지만 제 생활에서 직접 쓸 가능성은 더 확인해 봐야 합니다. "
                    f"{issue}의 이용 조건과 대상이 구체적으로 안내되면 판단하기 쉬울 것 같습니다."
                )
                concession = f"{issue}의 시범 운영 결과와 이용 기준이 공개되면 긍정적으로 검토할 수 있습니다."
            responses.append(
                {
                    "id": agent.persona_id,
                    "조정의견": natural_polite(opinion),
                    "양보조건": natural_polite(concession),
                }
            )
        summary = f"{issue}은 {low.persona_id if low else '우려 대표'}의 우려와 {high.persona_id if high else '기대 대표'}의 기대가 만나는 조정 쟁점입니다."
        return {
            "쟁점": issue,
            "사회자질문": moderator_question,
            "대표응답": responses,
            "조정정리": summary,
        }

    def final_proposal(
        self,
        agent: RepresentativeAgent,
        policy_text: str,
        related_answer: dict[str, Any] | None = None,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        keyword = agent.summary_keyword or "정책 보완"
        tags = set(agent.resource_tags)
        if "이용권운영" in tags:
            demand = "이용 목적과 실제 이용 빈도에 따라 지급량, 사용기한, 미사용분 처리 기준을 점검"
            concession = "일괄 확대가 어렵다면 병원 방문 등 필수 이동 수요가 큰 이용자부터 추가 지원을 검토하면 수용"
        elif "기사참여" in tags:
            demand = "기사의 호출 수락과 지역별 배차 실패를 점검하고 참여 유인을 포함한 운영 기준을 마련"
            concession = "전 지역 동시 개선이 어렵다면 배차 실패가 잦은 지역과 시간대부터 시범 점검하면 수용"
        elif "결제정산" in tags:
            demand = "이용자 결제와 기사 정산 절차를 단순화하고 처리 상태를 확인할 수 있게 운영"
            concession = "전면 개편 전 카드결제와 간소화된 정산 절차를 일부 지역에서 먼저 검토하면 수용"
        elif "동승운영" in tags:
            demand = "가족·마을 주민 동승과 공동배차의 이용 기준, 부담금, 예약 방식을 명확히 안내"
            concession = "안전과 정산 기준이 분명하다면 수요가 겹치는 시간대부터 공동배차를 검토하면 수용"
        elif "재정지속성" in tags:
            demand = "지원 확대에 필요한 재원과 군비 부담, 이용 성과를 함께 공개하고 지속 가능한 범위를 설정"
            concession = "효과와 수요가 확인되는 대상부터 단계적으로 확대하면 수용"
        elif "택시산업" in tags:
            demand = "택시업계 참여가 서비스 안정성과 지역 운행 수요로 이어지는지 함께 점검"
            concession = "서비스 기준을 지키는 참여 차량부터 인센티브와 협약 개선을 검토하면 수용"
        elif "생활이동" in tags:
            demand = "병원, 시장, 행정기관 등 필수 생활 목적별 이용 시간과 승하차 수요를 운영에 반영"
            concession = "모든 목적 확대가 어렵다면 의료와 장날 등 수요가 큰 이동부터 우선하면 수용"
        elif "교통" in tags:
            demand = "노선, 배차, 정류장 접근성, 이동 취약지역을 실제 이용자 기준으로 조정"
            concession = "전 지역 동시 확대가 어렵다면 교통 취약 지역과 이용 시간대부터 우선 적용하면 수용"
        elif "보건의료" in tags:
            demand = "고령자와 읍면 지역 주민이 이용 가능한 진료, 이동, 예약 지원을 함께 설계"
            concession = "상시 운영이 어렵다면 정기 방문, 예약 지원, 보건소 연계부터 시작하면 수용"
        elif "농업" in tags:
            demand = "농가 규모, 작목, 고령 농민의 현실을 구분해 장비, 교육, 판로 지원을 설계"
            concession = "전체 농가 확대 전 시범 작목이나 취약 농가부터 적용하면 수용"
        elif "관광문화" in tags:
            demand = "주민 생활 불편, 지역상권 효과, 방문객 동선을 함께 고려해 운영 계획을 조정"
            concession = "행사 규모를 단계적으로 키우되 주민 불편 관리 기준이 있으면 수용"
        elif "교육" in tags:
            demand = "연령대, 접근성, 수업 시간, 디지털 역량 차이를 반영한 교육 운영 방식 마련"
            concession = "전체 과정을 한 번에 늘리기 어렵다면 수요가 큰 대상부터 우선 개설하면 수용"
        elif "돌봄복지" in tags:
            demand = "취약계층별 신청, 이동, 이용 절차를 단순화하고 실제 돌봄 공백을 줄이는 방식으로 설계"
            concession = "보편 지원이 어렵다면 고위험 가구와 서비스 사각지대부터 우선 지원하면 수용"
        elif "지역경제" in tags:
            demand = "소상공인, 구직자, 지역기업의 체감 효과를 나눠 보고 지원 경로를 구체화"
            concession = "예산이 제한되면 효과 측정이 가능한 상권이나 업종부터 시범 운영하면 수용"
        elif "환경" in tags:
            demand = "환경 개선 효과와 주민 생활 불편을 함께 측정하고 관리 기준을 공개"
            concession = "전면 시행이 어렵다면 오염·불편이 큰 구역부터 우선 적용하면 수용"
        elif "주거정주" in tags:
            demand = "주거, 생활편의, 교통, 일자리 접근성을 함께 묶어 정주 여건 개선 효과를 높임"
            concession = "전체 지역 확대 전 인구 유출이나 빈집 문제가 큰 지역부터 시작하면 수용"
        elif "안전재난" in tags:
            demand = "위험지역, 취약시간대, 취약계층을 기준으로 예방과 대응 체계를 구체화"
            concession = "전면 구축이 어렵다면 사고 위험이 높은 지점부터 우선 적용하면 수용"
        elif "접근성" in tags:
            demand = "읍 외 지역도 이용할 수 있도록 출장 상담, 온라인 상담, 이동 지원을 함께 설계"
            concession = "센터 본관은 화순읍에 두되 보조 채널을 병행하면 수용"
        elif "대상적합성" in tags:
            demand = "청년뿐 아니라 중장년, 고령자, 경력단절자별 이용 시나리오를 분리"
            concession = "주 대상은 취업 희망자로 두되 안내와 연계 범위를 넓히면 수용"
        elif "비용형평성" in tags:
            demand = "예산 투입의 우선순위와 취약집단 체감 혜택을 보고서에 공개"
            concession = "효과가 확인되는 범위부터 단계적으로 확대하면 수용"
        elif "디지털" in tags:
            demand = "디지털 상담과 오프라인 상담을 동시에 운영해 정보 격차를 줄임"
            concession = "온라인 중심 운영은 가능하지만 오프라인 보완이 필요"
        elif "운영품질" in tags:
            demand = "담당 기관, 운영 절차, 성과 확인 방식, 민원 대응 창구를 명확히 공개"
            concession = "초기에는 간단한 운영 기준부터 공개하고 이용자 피드백으로 보완하면 수용"
        elif "취업지원" in tags:
            demand = "상담, 훈련, 채용 연계를 한 번에 확인할 수 있는 맞춤형 경로 제공"
            concession = "모든 직종을 한 번에 다루기 어렵다면 수요가 큰 분야부터 시작"
        else:
            demand = f"{keyword}에 대한 시민별 체감 차이를 정책 세부 설계에 반영"
            concession = "핵심 대상부터 시행하되 사각지대 점검을 병행하면 수용"

        if related_answer and related_answer.get("답변유형") in {"조건부수용", "부분수용"}:
            final_position = f"{keyword} 조건부 보완 수용"
        elif agent.satisfaction_score <= 2:
            final_position = f"{keyword} 보완 전제 유보"
        elif agent.satisfaction_score >= 4:
            final_position = f"{keyword} 기대 유지"
        else:
            final_position = f"{keyword} 조건부 검토"

        return {
            "id": agent.persona_id,
            "발언유형": "최종요구",
            "키워드": keyword,
            "최종입장": final_position,
            "핵심요구": demand,
            "양보가능지점": concession,
        }

    def moderator_summary(
        self,
        policy_text: str,
        agents: list[RepresentativeAgent],
        openings: list[dict[str, Any]],
        answers: list[dict[str, Any]],
        mediations: list[dict[str, Any]],
        final_proposals: list[dict[str, Any]],
        followups: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # 사회자는 '판단'이 아니라 공무원용 요약만 한다: 기존 발언을 재구성할 뿐, 새 사실/정책을 만들지 않는다.
        followups = followups or []
        lookup = {agent.persona_id: agent for agent in agents}
        stance_lines: list[str] = []
        for opening in openings:
            agent = lookup.get(opening.get("id"))
            label = persona_label(agent) if agent else clean_space(opening.get("id"))
            position = clean_space(opening.get("입장") or opening.get("claim"))
            if position:
                stance_lines.append(f"{label}: {position}")
        clashes: list[str] = []
        for item in followups:
            question = clean_space(item.get("재질문"))
            if question:
                clashes.append(f"{item.get('id')}→{item.get('지목_대상')}: {question}")
        issues = [clean_space(item.get("쟁점")) for item in mediations if clean_space(item.get("쟁점"))]
        demands = list(
            dict.fromkeys(
                [clean_space(item.get("핵심요구")) for item in final_proposals if clean_space(item.get("핵심요구"))]
                + [clean_space(item.get("보완요구")) for item in followups if clean_space(item.get("보완요구"))]
            )
        )
        return {
            "정책요지": policy_title(policy_text),
            "대표입장요약": stance_lines or [UNVERIFIED_TEXT],
            "핵심쟁점": clashes or issues or [UNVERIFIED_TEXT],
            "주민보완요구": demands or [UNVERIFIED_TEXT],
            "미해결쟁점": issues or [UNVERIFIED_TEXT],
            "한줄브리핑": (
                f"'{policy_title(policy_text)}'에 대해 대표 {len(agents)}명이 직접 수혜 여부 등에서 입장이 갈렸고, "
                f"주민 보완요구 {len(demands)}건이 도출되었습니다."
            ),
        }


class OpenAIDebateClient:
    OPENING_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "발언유형": {"type": "string", "enum": ["입론"]},
            "입장": {"type": "string"},
            "만족도": {"type": "integer", "minimum": 1, "maximum": 5},
            "claim": {"type": "string"},
            "ground": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "내용": {"type": "string"},
                    "인용_fact_id": {"type": "string"},
                },
                "required": ["내용", "인용_fact_id"],
            },
            "warrant": {"type": "string"},
            "핵심근거": {"type": "string"},
        },
        "required": ["id", "발언유형", "입장", "만족도", "claim", "ground", "warrant", "핵심근거"],
    }
    QUESTION_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "from_id": {"type": "string"},
            "to_id": {"type": "string"},
            "발언유형": {"type": "string", "enum": ["반박"]},
            "지목_대상": {"type": "string"},
            "상대발언인용": {"type": "string"},
            "쟁점": {"type": "string"},
            "claim": {"type": "string"},
            "ground": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "내용": {"type": "string"},
                    "인용_fact_id": {"type": "string"},
                },
                "required": ["내용", "인용_fact_id"],
            },
            "warrant": {"type": "string"},
            "질문": {"type": "string"},
        },
        "required": ["from_id", "to_id", "발언유형", "지목_대상", "상대발언인용", "쟁점", "claim", "ground", "warrant", "질문"],
    }
    ANSWER_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "발언유형": {"type": "string", "enum": ["답변"]},
            "지목_대상": {"type": "string"},
            "답변유형": {"type": "string", "enum": list(ANSWER_TYPE_ENUM)},
            "답변": {"type": "string"},
        },
        "required": ["id", "발언유형", "지목_대상", "답변유형", "답변"],
    }
    FOLLOWUP_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "발언유형": {"type": "string", "enum": ["재질문"]},
            "지목_대상": {"type": "string"},
            "수용여부": {"type": "string", "enum": list(FOLLOWUP_ACCEPT_ENUM)},
            "재질문": {"type": "string"},
            "보완요구": {"type": "string"},
        },
        "required": ["id", "발언유형", "지목_대상", "수용여부", "재질문", "보완요구"],
    }
    MEDIATION_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "쟁점": {"type": "string"},
            "사회자질문": {"type": "string"},
            "대표응답": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "조정의견": {"type": "string"},
                        "양보조건": {"type": "string"},
                    },
                    "required": ["id", "조정의견", "양보조건"],
                },
            },
            "조정정리": {"type": "string"},
        },
        "required": ["쟁점", "사회자질문", "대표응답", "조정정리"],
    }
    FINAL_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "발언유형": {"type": "string", "enum": ["최종요구"]},
            "키워드": {"type": "string"},
            "최종입장": {"type": "string"},
            "핵심요구": {"type": "string"},
            "양보가능지점": {"type": "string"},
        },
        "required": ["id", "발언유형", "키워드", "최종입장", "핵심요구", "양보가능지점"],
    }
    MODERATOR_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "정책요지": {"type": "string"},
            "대표입장요약": {"type": "array", "items": {"type": "string"}},
            "핵심쟁점": {"type": "array", "items": {"type": "string"}},
            "주민보완요구": {"type": "array", "items": {"type": "string"}},
            "미해결쟁점": {"type": "array", "items": {"type": "string"}},
            "한줄브리핑": {"type": "string"},
        },
        "required": ["정책요지", "대표입장요약", "핵심쟁점", "주민보완요구", "미해결쟁점", "한줄브리핑"],
    }

    SYSTEM_PROMPT = """당신은 전남 화순군 정책 시뮬레이션의 대표 시민 토론을 진행·정리하는 중립 조정 agent입니다.
대표 시민은 실제 토론 참여자이고, 사회자는 시민 페르소나가 아닙니다.
모든 시민 발언은 공론장에서 실제 주민이 서로 예의를 지키며 말하는 자연스러운 존댓말 해요체로 작성하십시오. 반말, 음슴체, 보고서식 명사 종결을 사용하지 마십시오.
각 대표는 자신의 원래 입장과 생활 조건을 유지해야 하며, 앞선 발언에서 말하지 않은 경험이나 주장을 나중에 본인이 이미 말한 것처럼 만들지 마십시오.
정책 원문, 대표 시민 속성, 기존 발언, 증거팩, 반대신문 답변과 최종 개선 요구를 사실 판단의 우선 근거로 사용하십시오.
정책 유형별 일반 점검 쟁점과 대표 시민의 생활 조건에서 논리적으로 이어지는 운영상 보완점은 입력에 직접 적혀 있지 않아도 '검토 요구', '우려', '조건부 제안' 형태로 한 단계 확장할 수 있습니다.
이때 실제 문제가 이미 발생했다거나 주민 다수가 요구했다고 단정하지 말고, '발생할 수 있으므로 확인이 필요하다', '이런 조건을 검토해 달라'처럼 가설임을 분명히 하십시오.
실제 주민 반응을 알고 있는 것처럼 말하지 말고, 확인되지 않은 여론 표현은 만들지 마십시오.
증거팩 카드에 없는 수치·사실은 만들어내지 말고 '토론 기록상 확인되지 않음'으로 처리하십시오.
증거가 없는 새 제안에는 구체적인 수치·고유명사·발생 사실을 붙이지 말고 일반적인 운영 조건만 제시하십시오.
찬성/반대 이분법보다 어떤 조건, 불편, 기대, 보완 요구가 있는지 구조화하십시오.
응답은 반드시 요청된 JSON 스키마만 따르고, 과장된 문장이나 없는 사실을 추가하지 마십시오."""

    def __init__(
        self,
        model: str,
        api_key_env: str,
        fallback: LocalDebateClient,
        strict_api: bool = False,
        warnings: list[str] | None = None,
    ):
        load_env_files()
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} 환경변수가 없습니다.")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("openai 패키지가 설치되어 있지 않습니다.") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fallback = fallback
        self.strict_api = strict_api
        self.warnings = warnings if warnings is not None else []

    def opening(self, agent: RepresentativeAgent, policy_text: str, evidence_pack: dict[str, Any] | None = None) -> dict[str, Any]:
        prompt = f"""정책:
{policy_text}

대표 시민:
{json.dumps(asdict(agent), ensure_ascii=False)}

배정 증거팩:
{format_evidence_pack(evidence_pack)}

위 대표 시민이 1인칭 존댓말로 2문장의 첫 발언을 한다고 보고 JSON을 작성하십시오.
첫 문장은 "저는 이 정책에 조건부로 찬성합니다"처럼 자신의 입장을 분명히 밝히십시오. 둘째 문장은 본인의 거주지·교통수단·가족·직업 등 실제 생활 조건에서 나온 이유를 말하십시오.
입장은 찬/반 단어 하나가 아니라 '접근성 보완을 전제로 찬성', '직접 혜택이 적어 유보'처럼 구체적으로 쓰십시오.
Claim-Ground-Warrant를 반드시 분리하고, Ground에는 배정 증거팩에 실제 존재하는 fact_id 하나를 인용하십시오.
대표 시민의 속성·만족도·읍면·태그와 모순되지 않게, 내 상황에서 잃을 수 있는 것과 체감 조건을 드러내십시오.
증거팩에 없는 수치나 사실은 '{UNVERIFIED_TEXT}'으로 쓰십시오. 모든 문장은 자연스러운 존댓말로 끝내십시오."""
        return self._call_or_fallback("opening_statement", self.OPENING_SCHEMA, prompt, self.fallback.opening, agent, policy_text, evidence_pack)

    def question(
        self,
        asker: RepresentativeAgent,
        target: RepresentativeAgent,
        pair: DebatePair,
        policy_text: str,
        target_opening: dict[str, Any] | None = None,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = f"""정책:
{policy_text}

질문자 대표:
{json.dumps(asdict(asker), ensure_ascii=False)}

답변자 대표:
{json.dumps(asdict(target), ensure_ascii=False)}

쌍 선정 이유:
{json.dumps(asdict(pair), ensure_ascii=False)}

상대의 입론:
{json.dumps(target_opening or {}, ensure_ascii=False)}

질문자 배정 증거팩:
{format_evidence_pack(evidence_pack)}

질문자는 답변자를 공격하지 말고, 자원 충돌 또는 체감 조건을 확인하는 존댓말 질문을 1문장으로 작성하십시오.
'상대발언인용'에는 상대 입론에서 실제로 한 문장을 글자 그대로 넣으십시오.
'질문'은 반드시 "앞서 ○○라고 말씀하셨는데"로 시작하고, ○○는 상대 입론에 실제로 등장한 핵심 표현이어야 합니다. 상대가 말하지 않은 배차 편의, 호출 성공, 이용 경험 등을 말했다고 가정하지 마십시오.
그 지점이 내 이해관계나 체감 조건과 왜 충돌하는지 claim-ground-warrant로 밝히고, Ground에는 질문자에게 배정된 증거팩에 실제 존재하는 fact_id 하나를 인용하십시오."""
        return self._call_or_fallback(
            "cross_exam_question",
            self.QUESTION_SCHEMA,
            prompt,
            self.fallback.question,
            asker,
            target,
            pair,
            policy_text,
            target_opening,
            evidence_pack,
        )

    def answer(
        self,
        target: RepresentativeAgent,
        asker: RepresentativeAgent,
        question: dict[str, Any],
        policy_text: str,
        target_opening: dict[str, Any] | None = None,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = f"""정책:
{policy_text}

답변자 대표:
{json.dumps(asdict(target), ensure_ascii=False)}

질문자 대표:
{json.dumps(asdict(asker), ensure_ascii=False)}

질문:
{json.dumps(question, ensure_ascii=False)}

답변자가 앞서 한 입론:
{json.dumps(target_opening or {}, ensure_ascii=False)}

답변자는 첫 문장에서 앞서 밝힌 자신의 입장을 짧게 다시 확인하고, 둘째 문장에서 질문에 직접 답하십시오.
앞선 입론에 없던 경험이나 주장을 새로 만들지 말고, 즉시 전면 수용하지 말며 원래 의견과 이해관계를 유지하십시오.
답변유형은 {', '.join(ANSWER_TYPE_ENUM)} 중 하나만 사용하십시오. '무조건수용' 또는 이와 같은 전면 수용 표현은 금지입니다.
답변은 자연스러운 존댓말 2문장 이내로 작성하십시오."""
        return self._call_or_fallback(
            "cross_exam_answer",
            self.ANSWER_SCHEMA,
            prompt,
            self.fallback.answer,
            target,
            asker,
            question,
            policy_text,
            target_opening,
            evidence_pack,
        )

    def followup(
        self,
        asker: RepresentativeAgent,
        target: RepresentativeAgent,
        question: dict[str, Any],
        answer: dict[str, Any],
        policy_text: str,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = f"""정책:
{policy_text}

질문자 대표:
{json.dumps(asdict(asker), ensure_ascii=False)}

앞서 내가 한 질문:
{json.dumps(question, ensure_ascii=False)}

상대의 답변:
{json.dumps(answer, ensure_ascii=False)}

질문자는 상대의 답변을 듣고 재반박을 1턴 더 합니다.
먼저 내 이해관계가 실제로 해소됐는지 '수용여부'로 판단하십시오('일부해소' 또는 '미해소' 중 하나).
그다음 답변에서 확인되지 않은 부분을 짚어, '누가/어떤 경우에 빠지는가'를 드러내는 재질문 1문장과,
정책에 넣어야 할 구체적 보완요구 1문장을 쓰십시오.
상대 답변을 길게 옮겨 적지 말고 핵심만 짧게 지목하십시오.
전면 수용하거나 감사 인사로 끝내지 말고, 새로운 사실·수치는 만들지 마십시오. 모든 문장은 자연스러운 존댓말로 작성하십시오."""
        return self._call_or_fallback(
            "cross_exam_followup",
            self.FOLLOWUP_SCHEMA,
            prompt,
            self.fallback.followup,
            asker,
            target,
            question,
            answer,
            policy_text,
            evidence_pack,
        )

    def issue_mediation(
        self,
        issue: str,
        agents: list[RepresentativeAgent],
        policy_text: str,
        answers: list[dict[str, Any]],
        evidence_packs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        prompt = f"""정책:
{policy_text}

조정 쟁점:
{issue}

참여 대표:
{json.dumps([asdict(agent) for agent in agents], ensure_ascii=False)}

반대신문 답변:
{json.dumps(answers, ensure_ascii=False)}

이 단계는 새로운 시민 의견을 만드는 단계가 아닙니다.
사회자는 이 쟁점에 대해 "어떤 조건이면 수용 가능한가"만 묻는 조정 질문을 1문장으로 던지십시오.
대표응답은 각 대표의 원래 속성, 만족도, 한 줄 의견, 이전 발언과 모순되지 않는 범위에서만 자연스러운 존댓말 1문장씩 작성하십시오.
같은 결론을 반복하지 말고, 직접 수혜 여부·교통비·병원 및 시장 접근·호출과 배차·대상 기준처럼 각 대표가 가진 서로 다른 관점이 드러나게 하십시오.
대표가 기존에 직접 말하지 않았더라도 조정 쟁점과 생활 조건에서 자연스럽게 이어지는 일반적인 운영 조건은 '검토 요구'로 1개까지 제안할 수 있습니다. 다만 실제 불만·피해가 확인됐다고 단정하거나 새 수치를 만들지는 마십시오.
사회자의 조정정리는 확인된 조건과 새 검토 요구를 구분해 묶어 작성하십시오.
찬반 표결이 아니라 실행 조건 조정에 집중하십시오."""
        return self._call_or_fallback(
            "issue_mediation",
            self.MEDIATION_SCHEMA,
            prompt,
            self.fallback.issue_mediation,
            issue,
            agents,
            policy_text,
            answers,
            evidence_packs,
        )

    def final_proposal(
        self,
        agent: RepresentativeAgent,
        policy_text: str,
        related_answer: dict[str, Any] | None = None,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = f"""정책:
{policy_text}

대표 시민:
{json.dumps(asdict(agent), ensure_ascii=False)}

관련 반대신문 답변:
{json.dumps(related_answer or {}, ensure_ascii=False)}

배정 증거팩:
{format_evidence_pack(evidence_pack)}

이 대표 시민의 최종 개선 요구를 JSON으로 작성하십시오.
핵심요구는 정책을 어떻게 고쳐야 하는지 바로 읽히게 쓰되, 대표 시민의 원래 의견과 관련 답변에서 근거가 확인되는 범위로 제한하십시오.
새로운 시민 집단을 임의로 만들지는 마십시오. 다만 정책 유형, 대표의 생활 조건, 토론 쟁점에서 한 단계로 추론되는 일반적인 운영 보완은 '검토 요구'로 제시할 수 있습니다.
증거가 없는 제안은 실제 발생 사실이나 주민 여론으로 단정하지 말고 구체적인 수치 없이 조건부로 표현하십시오."""
        return self._call_or_fallback(
            "final_proposal",
            self.FINAL_SCHEMA,
            prompt,
            self.fallback.final_proposal,
            agent,
            policy_text,
            related_answer,
            evidence_pack,
        )

    def moderator_summary(
        self,
        policy_text: str,
        agents: list[RepresentativeAgent],
        openings: list[dict[str, Any]],
        answers: list[dict[str, Any]],
        mediations: list[dict[str, Any]],
        final_proposals: list[dict[str, Any]],
        followups: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        followups = followups or []
        prompt = f"""정책:
{policy_text}

대표 시민:
{json.dumps([asdict(agent) for agent in agents], ensure_ascii=False)}

대표 발언:
{json.dumps(openings, ensure_ascii=False)}

반대신문 답변:
{json.dumps(answers, ensure_ascii=False)}

재질문·보완요구:
{json.dumps(followups, ensure_ascii=False)}

쟁점별 조정 라운드:
{json.dumps(mediations, ensure_ascii=False)}

최종 개선 요구:
{json.dumps(final_proposals, ensure_ascii=False)}

당신은 사회자입니다. 이 토론을 '바쁜 담당 공무원이 30초 안에 파악'하도록 요약만 하십시오.
합의를 도출하는 것이 목적이 아니므로 억지로 합의점을 만들지 마십시오.
- 정책요지: 무슨 정책인지 1문장.
- 대표입장요약: 대표별로 '누구(속성) — 입장' 형태로 1줄씩.
- 핵심쟁점: 반대신문·재질문에서 실제로 부딪힌 지점만 1줄씩.
- 주민보완요구: 최종 개선 요구와 재질문의 보완요구에서 확인된 구체적 보완점만 1줄씩.
- 미해결쟁점: 답이 나오지 않은 쟁점.
- 한줄브리핑: 전체를 한 문장으로.
위 입력에 없는 정책 효과, 여론 규모, 새 시민 요구는 추정하지 말고, 근거 없는 항목은 넣지 마십시오."""
        return self._call_or_fallback(
            "moderator_summary",
            self.MODERATOR_SCHEMA,
            prompt,
            self.fallback.moderator_summary,
            policy_text,
            agents,
            openings,
            answers,
            mediations,
            final_proposals,
            followups,
        )

    def _call_or_fallback(
        self,
        schema_name: str,
        schema: dict[str, Any],
        prompt: str,
        fallback_method: Any,
        *fallback_args: Any,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return self._call_json(schema_name, schema, prompt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.warnings.append(
                    f"api_retry:{schema_name}:{attempt + 1}/2:{type(exc).__name__}:{exc}"
                )
        if self.strict_api:
            raise RuntimeError(f"{schema_name} API 응답을 2회 파싱하지 못했습니다: {last_error}") from last_error
        self.warnings.append(f"fallback:{schema_name}:{type(last_error).__name__}:{last_error}")
        return fallback_method(*fallback_args)

    def _call_json(self, schema_name: str, schema: dict[str, Any], prompt: str) -> dict[str, Any]:
        token_budget = {
            "opening_statement": 420,
            "cross_exam_question": 460,
            "cross_exam_answer": 300,
            "cross_exam_followup": 340,
            "issue_mediation": 900,
            "final_proposal": 520,
            "moderator_summary": 900,
        }.get(schema_name, 420)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=0.9,
            max_output_tokens=token_budget,
        )
        content = getattr(response, "output_text", None) or self._extract_response_text(response)
        return self._loads_json(content)

    def _extract_response_text(self, response: Any) -> str:
        output = getattr(response, "output", None) or []
        chunks: list[str] = []
        for item in output:
            contents = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", []) or []
            for content in contents:
                text = content.get("text") if isinstance(content, dict) else getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    def _loads_json(self, content: str | None) -> dict[str, Any]:
        if not content:
            raise ValueError("API 응답 텍스트가 비어 있습니다.")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))


def build_debate_client(args: argparse.Namespace, warnings: list[str]) -> LocalDebateClient | OpenAIDebateClient:
    fallback = LocalDebateClient()
    if args.no_api or args.api_mode == "local":
        return fallback
    try:
        return OpenAIDebateClient(
            model=args.model,
            api_key_env=args.api_key_env,
            fallback=fallback,
            strict_api=args.api_mode == "api",
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001
        if args.api_mode == "api":
            raise
        warnings.append(f"API 모드 초기화가 어려워 로컬 규칙 기반 토론으로 진행했습니다: {exc}")
        return fallback


def select_mediation_issues(
    agents: list[RepresentativeAgent],
    policy_text: str,
    max_issues: int = 6,
    max_observed_issues: int = 3,
) -> list[str]:
    """조정 쟁점을 고른다.

    토픽 드리프트 방지(Part C): 대표들이 짚은 자원 태그를 먼저 넣으면, 비수혜 대표가
    많을 때 '고령층 접근성'처럼 정책 핵심과 어긋난 주제로 토론이 쏠린다. 그래서
    '정책 자체의 핵심 쟁점'(카테고리 템플릿)을 앵커로 먼저 깔고, 현장에서 관측된
    쟁점을 번갈아 섞어 정책 중심을 유지한다.
    """
    category = infer_policy_category(policy_text)
    core_issues = list(POLICY_ISSUE_TEMPLATES.get(category, POLICY_ISSUE_TEMPLATES["일반"]))

    counter: Counter[str] = Counter()
    for agent in agents:
        for tag in agent.resource_tags:
            if tag != "기타":
                counter[tag] += 1
    if not counter:
        for agent in agents:
            if agent.summary_keyword:
                counter[agent.summary_keyword] += 1
    observed_issues = [issue for issue, _ in counter.most_common(max_observed_issues)]

    # 핵심 쟁점 먼저, 관측 쟁점을 번갈아(core, observed, core, observed …) 배치.
    interleaved: list[str] = []
    for core, observed in zip_longest(core_issues, observed_issues):
        if core:
            interleaved.append(core)
        if observed:
            interleaved.append(observed)

    selected: list[str] = []
    for issue in interleaved:
        if not issues_overlap(issue, selected):
            selected.append(issue)
        if len(selected) >= max_issues:
            break
    return selected


def has_interest_difference(agents: list[RepresentativeAgent]) -> bool:
    if len(agents) < 2:
        return False
    profiles = {
        (
            "direct_benefit" if any(term in agent.opinion_text for term in ("아이", "자식", "학생", "우리 집 아이")) else "indirect_benefit",
            "high" if agent.satisfaction_score >= 4 else "middle" if agent.satisfaction_score >= 3 else "low",
            agent.summary_keyword,
        )
        for agent in agents
    }
    return len(profiles) >= 2


def call_with_verification(
    make_statement: Any,
    validate: Any,
    warnings: list[str],
    label: str,
) -> dict[str, Any]:
    statement = make_statement()
    result = validate(statement)
    if result["passed"]:
        return attach_verification(statement, result)
    warnings.append(f"검증 실패 후 1회 재생성:{label}:{result['failures']}")
    regenerated = make_statement()
    regenerated_result = validate(regenerated)
    if not regenerated_result["passed"]:
        warnings.append(f"검증 재실패 플래그 유지:{label}:{regenerated_result['failures']}")
    return attach_verification(regenerated, regenerated_result, regenerated=True)


def run_debate(
    agents: list[RepresentativeAgent],
    policy_text: str,
    client: LocalDebateClient | OpenAIDebateClient,
    max_cross_exams: int,
    evidence_packs: dict[str, dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    evidence_packs = evidence_packs or {}
    warnings = warnings if warnings is not None else []
    openings: list[dict[str, Any]] = []
    for agent in agents:
        opening = call_with_verification(
            lambda agent=agent: client.opening(agent, policy_text, evidence_packs.get(agent.persona_id)),
            lambda statement, agent=agent: validate_statement(
                statement,
                stage="opening",
                pack=evidence_packs.get(agent.persona_id),
                expected_stance=agent.stance or score_position(agent.satisfaction_score, agent.summary_keyword),
            ),
            warnings,
            f"opening:{agent.persona_id}",
        )
        verification_failures = ((opening.get("_verification") or {}).get("failures") or [])
        if any(str(item).startswith("stance_mismatch:") for item in verification_failures):
            # 재생성 후에도 기존 입장을 뒤집으면 그 결과를 보고서에 쓰지 않는다.
            warnings.append(f"입장 불일치 AI 입론 폐기 후 기존 입장 유지:{agent.persona_id}")
            safe_opening = LocalDebateClient().opening(
                agent, policy_text, evidence_packs.get(agent.persona_id)
            )
            safe_result = validate_statement(
                safe_opening,
                stage="opening",
                pack=evidence_packs.get(agent.persona_id),
                expected_stance=agent.stance or score_position(agent.satisfaction_score, agent.summary_keyword),
            )
            opening = attach_verification(safe_opening, safe_result, regenerated=True)
        # AI가 HW035_1st처럼 임의 접미사를 붙여도 대화 기억이 끊기지 않도록 고정한다.
        opening["id"] = agent.persona_id
        openings.append(opening)
    pairs = select_debate_pairs(agents, max_cross_exams)
    agent_lookup = {agent.persona_id: agent for agent in agents}
    opening_by_agent = {_canonical_pid(opening.get("id", "")): opening for opening in openings}

    questions: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    followups: list[dict[str, Any]] = []
    answer_by_agent: dict[str, dict[str, Any]] = {}

    for pair in pairs:
        asker = agent_lookup[pair.asker_id]
        target = agent_lookup[pair.target_id]
        target_opening = opening_by_agent.get(target.persona_id) or {}
        quote_source = visible_opening_text(target_opening)
        question = call_with_verification(
            lambda asker=asker, target=target, pair=pair, target_opening=target_opening: ground_question_in_visible_opening(
                client.question(
                    asker,
                    target,
                    pair,
                    policy_text,
                    target_opening,
                    evidence_packs.get(asker.persona_id),
                ),
                target_opening,
            ),
            lambda statement, asker=asker, quote_source=quote_source: validate_statement(
                statement,
                stage="rebuttal",
                pack=evidence_packs.get(asker.persona_id),
                target_quote_source=quote_source,
            ),
            warnings,
            f"rebuttal:{asker.persona_id}->{target.persona_id}",
        )
        answer = call_with_verification(
            lambda target=target, asker=asker, question=question, target_opening=target_opening: client.answer(
                target,
                asker,
                question,
                policy_text,
                target_opening,
                evidence_packs.get(target.persona_id),
            ),
            lambda statement, target_opening=target_opening: validate_statement(
                statement,
                stage="answer",
                target_quote_source=visible_opening_text(target_opening),
            ),
            warnings,
            f"answer:{target.persona_id}",
        )
        followup = call_with_verification(
            lambda asker=asker, target=target, question=question, answer=answer: client.followup(
                asker,
                target,
                question,
                answer,
                policy_text,
                evidence_packs.get(asker.persona_id),
            ),
            lambda statement: validate_statement(statement, stage="followup"),
            warnings,
            f"followup:{asker.persona_id}->{target.persona_id}",
        )
        questions.append({"pair_id": pair.pair_id, **question})
        answers.append({"pair_id": pair.pair_id, "question_to": pair.target_id, **answer})
        followups.append({"pair_id": pair.pair_id, "question_to": pair.target_id, **followup})
        answer_by_agent[pair.target_id] = answer

    mediations: list[dict[str, Any]] = []
    # 대표 키워드와 정책 핵심 쟁점을 고르게 다뤄 한두 주제에 토론이 쏠리지 않게 한다.
    for issue in select_mediation_issues(agents, policy_text, max_issues=4, max_observed_issues=4):
        issue_tags = [tag for tag in extract_resource_tags(issue) if tag != "기타"]
        issue_agents = [
            agent
            for agent in agents
            if issue in agent.resource_tags
            or issue == agent.summary_keyword
            or any(tag in agent.resource_tags for tag in issue_tags)
        ] or agents
        if not has_interest_difference(issue_agents):
            warnings.append(f"조정 라운드 생략:{issue}:이해관계 차이 대표 2명 미만")
            continue
        mediations.append(client.issue_mediation(issue, issue_agents, policy_text, answers, evidence_packs))

    final_proposals = [
        call_with_verification(
            lambda agent=agent: client.final_proposal(
                agent,
                policy_text,
                answer_by_agent.get(agent.persona_id),
                evidence_packs.get(agent.persona_id),
            ),
            lambda statement: validate_statement(statement, stage="final"),
            warnings,
            f"final:{agent.persona_id}",
        )
        for agent in agents
    ]
    # Fishkin-style facilitator: 사회자는 판단·합의 도출이 아니라 공무원용 요약만 한다.
    # 합의를 강제하지 않고, 미해결 쟁점은 실패가 아니라 정책 보완을 위한 갈등 지도로 남긴다.
    moderator_summary = client.moderator_summary(
        policy_text,
        agents,
        openings,
        answers,
        mediations,
        final_proposals,
        followups,
    )
    verification_quality = validation_metrics([*openings, *questions, *answers, *followups, *final_proposals])

    return {
        "openings": openings,
        "pairs": [asdict(pair) for pair in pairs],
        "questions": questions,
        "answers": answers,
        "followups": followups,
        "mediations": mediations,
        "final_proposals": final_proposals,
        "moderator_summary": moderator_summary,
        "verification_quality": verification_quality,
        "evidence_packs": evidence_packs,
    }


def policy_to_text(policy: dict[str, Any], responses: list[CitizenResponse]) -> str:
    if policy:
        policy_name = clean_space(policy.get("policy_name") or policy.get("name") or "")
        main_details = clean_space(policy.get("main_details") or policy.get("details") or "")
        target_audience = clean_space(policy.get("target_audience") or "")
        keywords = policy.get("search_keywords") or []
        keyword_text = ", ".join(str(keyword) for keyword in keywords) if isinstance(keywords, list) else clean_space(keywords)
        parts = [part for part in (policy_name, main_details, target_audience, keyword_text) if part]
        if parts:
            return " / ".join(parts)
    return responses[0].policy_id or "정책명 미상"


def policy_title(policy_text: str) -> str:
    text = clean_space(policy_text)
    for marker in (" / 기간", "/ 기간", " / 장소", "/ 장소"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    text = re.sub(r"^정책명\s*:\s*", "", text)
    return shorten(text, 80) or "정책명 미상"


def satisfaction_counts(responses: list[CitizenResponse]) -> dict[int, int]:
    counts = {score: 0 for score in range(1, 6)}
    for response in responses:
        score = int(round(response.satisfaction_score))
        score = min(5, max(1, score))
        counts[score] += 1
    return counts


def satisfaction_summary(responses: list[CitizenResponse]) -> dict[str, Any]:
    counts = satisfaction_counts(responses)
    total = sum(counts.values())
    average = round(
        sum(score * count for score, count in counts.items()) / total,
        3,
    ) if total else 0.0
    return {
        "total": total,
        "average": average,
        "counts": counts,
        "generated_ratios": {
            score: round((count / total * 100), 1) if total else 0.0
            for score, count in counts.items()
        },
    }


def satisfaction_block(summary: dict[str, Any]) -> str:
    total = int(summary["total"])
    counts: dict[int, int] = {int(score): int(count) for score, count in summary["counts"].items()}
    generated_ratios: dict[int, float] = {
        int(score): float(ratio)
        for score, ratio in summary["generated_ratios"].items()
    }

    lines: list[str] = []
    for score in range(1, 6):
        count = counts.get(score, 0)
        ratio = generated_ratios.get(score, 0.0)
        bar = "█" * count if count > 0 else ""
        lines.append(
            f"{score}점 {SCORE_LABELS[score]:<7} | {bar:<50} {count:3d}명 ({ratio:5.1f}%)"
        )
    lines.append("----------------------------------------------")
    lines.append(f"평균 만족도: {summary['average']:.2f} / 5.0  (총 {total}명)")
    return "\n".join(lines)


def top_keywords(responses: list[CitizenResponse], limit: int = 5) -> list[tuple[str, int]]:
    counter: dict[str, int] = {}
    for response in responses:
        keyword = response.summary_keyword or "키워드 없음"
        counter[keyword] = counter.get(keyword, 0) + 1
    return sorted(counter.items(), key=lambda item: item[1], reverse=True)[:limit]


def build_one_page_summary(
    policy_text: str,
    representatives: list[RepresentativeAgent],
    debate: dict[str, Any],
    responses: list[CitizenResponse],
    satisfaction: dict[str, Any],
) -> str:
    keywords = ", ".join(f"{keyword}({count})" for keyword, count in top_keywords(responses))
    pair_lines = []
    for pair in debate["pairs"]:
        pair_lines.append(
            f"- {pair['asker_id']} -> {pair['target_id']}: {pair['reason']} "
            f"(쌍 점수 {pair['pair_score']})"
        )
    if not pair_lines:
        pair_lines.append("- 대표 agent 수가 적어 별도 반대신문 쌍을 만들지 않았습니다.")

    mediation_lines = []
    for mediation in debate.get("mediations", []):
        mediation_lines.append(
            f"- {mediation.get('쟁점')}: {mediation.get('조정정리')}"
        )
    if not mediation_lines:
        mediation_lines.append("- 별도 조정 라운드 없음")

    moderator = debate.get("moderator_summary") or {}
    moderator_lines = []
    brief = clean_space(moderator.get("한줄브리핑"))
    if brief:
        moderator_lines.append(f"- 한줄 브리핑: {brief}")
    for item in moderator.get("주민보완요구", [])[:5]:
        moderator_lines.append(f"- 주민 보완요구: {item}")
    for item in moderator.get("미해결쟁점", [])[:3]:
        moderator_lines.append(f"- 미해결 쟁점: {item}")
    if not moderator_lines:
        moderator_lines.append("- 최종 개선 요구를 기준으로 정책 보완 필요")

    improvement_lines = []
    seen_demands: set[str] = set()
    for proposal in debate["final_proposals"]:
        demand = clean_space(proposal.get("핵심요구"))
        if demand and demand not in seen_demands:
            improvement_lines.append(f"- {proposal.get('id')}: {demand}")
            seen_demands.add(demand)

    rep_lines = [
        f"- {persona_label(agent)} | {agent.summary_keyword} | {agent.satisfaction_score:g}점 | {agent.selection_reason or agent.cluster_name}"
        for agent in representatives
    ]

    return "\n".join(
        [
            "## 1쪽 요약",
            "",
            f"- 정책: {policy_title(policy_text)}",
            f"- 평균 만족도: {satisfaction['average']:.2f}/5.0, 응답자 {satisfaction['total']}명",
            f"- 주요 의견 키워드: {keywords}",
            "",
            "### 토론 대표 agent",
            *rep_lines,
            "",
            "### 반대신문으로 확인한 충돌 지점",
            *pair_lines,
            "",
            "### 쟁점별 조정 라운드",
            *mediation_lines,
            "",
            "### 사회자 브리핑 (공무원용)",
            *moderator_lines,
            "",
            "### 정책 개선 요구",
            *improvement_lines,
        ]
    )


def format_all_opinions(responses: list[CitizenResponse]) -> str:
    def sort_key(response: CitizenResponse) -> tuple[str, str]:
        match = re.search(r"(\d+)", response.persona_id)
        number = f"{int(match.group(1)):04d}" if match else response.persona_id
        return number, response.persona_id

    lines = ["## 100명 시민 의견", ""]
    for response in sorted(responses, key=sort_key):
        age = f"{response.age}세" if response.age is not None else "연령미상"
        lines.append(
            f"- {response.persona_id} | {response.region_name_kr or '지역미상'} | {age} | "
            f"{response.satisfaction_score:g}점 | {response.summary_keyword or '키워드 없음'} | {response.opinion_text}"
        )
    return "\n".join(lines)


def format_debate_transcript(
    representatives: list[RepresentativeAgent],
    debate: dict[str, Any],
) -> str:
    agent_lookup = {agent.persona_id: agent for agent in representatives}
    lines: list[str] = ["## 토론 전문", ""]

    lines.extend(["### 1단계. 대표 발언", ""])
    for opening in debate["openings"]:
        agent = agent_lookup.get(opening.get("id"))
        label = persona_label(agent) if agent else opening.get("id", "")
        lines.append(f"- {label}: {json.dumps(opening, ensure_ascii=False)}")

    lines.extend(["", "### 2단계. 반대신문 질문", ""])
    if debate["questions"]:
        for question in debate["questions"]:
            lines.append(f"- {question.get('pair_id')}: {json.dumps(question, ensure_ascii=False)}")
    else:
        lines.append("- 반대신문 질문 없음")

    lines.extend(["", "### 3단계. 답변 및 입장 보완", ""])
    if debate["answers"]:
        for answer in debate["answers"]:
            lines.append(f"- {answer.get('pair_id')}: {json.dumps(answer, ensure_ascii=False)}")
    else:
        lines.append("- 답변 없음")

    lines.extend(["", "### 3-2단계. 재질문·보완요구 (재반박)", ""])
    if debate.get("followups"):
        for followup in debate["followups"]:
            lines.append(f"- {followup.get('pair_id')}: {json.dumps(followup, ensure_ascii=False)}")
    else:
        lines.append("- 재질문 없음")

    lines.extend(["", "### 4단계. 쟁점별 조정 라운드", ""])
    if debate.get("mediations"):
        for mediation in debate["mediations"]:
            lines.append(f"- {mediation.get('쟁점')}: {json.dumps(mediation, ensure_ascii=False)}")
    else:
        lines.append("- 조정 라운드 없음")

    lines.extend(["", "### 5단계. 최종 개선 요구", ""])
    for proposal in debate["final_proposals"]:
        lines.append(f"- {proposal.get('id')}: {json.dumps(proposal, ensure_ascii=False)}")

    lines.extend(["", "### 6단계. 사회자 브리핑 (공무원용)", ""])
    moderator = debate.get("moderator_summary") or {}
    policy_gist = clean_space(moderator.get("정책요지"))
    if policy_gist:
        lines.append(f"- 정책 요지: {policy_gist}")
    brief = clean_space(moderator.get("한줄브리핑"))
    if brief:
        lines.append(f"- 한줄 브리핑: {brief}")

    section_labels = [
        ("대표입장요약", "대표 입장"),
        ("핵심쟁점", "핵심 쟁점"),
        ("주민보완요구", "주민 보완요구"),
        ("미해결쟁점", "미해결 쟁점"),
    ]
    for key, label in section_labels:
        items = [clean_space(item) for item in moderator.get(key, []) if clean_space(item)]
        if not items:
            continue
        lines.append(f"- {label}:")
        lines.extend(f"    - {item}" for item in items)

    return "\n".join(lines)


def format_technology_pipeline(cluster_run_summary: dict[str, Any]) -> list[str]:
    feature_meta = cluster_run_summary.get("feature_metadata") or {}
    reducer_meta = cluster_run_summary.get("reducer_metadata") or {}
    tech = cluster_run_summary.get("technology_pipeline") or {}
    keyword_backend = feature_meta.get("keyword_backend") or "unknown"
    opinion_backend = feature_meta.get("opinion_backend") or "unknown"
    reducer = reducer_meta.get("reducer") or "unknown"
    algorithm = cluster_run_summary.get("algorithm") or "unknown"
    return [
        "## 기술 파이프라인 요약",
        "",
        f"- 시민 의견 생성: {tech.get('citizen_response_generation', '외부 팀 코드')}",
        f"- 의견 벡터화: keyword={keyword_backend}, opinion={opinion_backend}",
        f"- OpenAI Embedding 모델: {feature_meta.get('openai_embedding_model') or '미사용'}",
        f"- SBERT 모델: {feature_meta.get('embedding_model') or '미사용'}",
        f"- 차원 축소: {reducer}"
        + (f" ({reducer_meta.get('input_dim')}차원 -> {reducer_meta.get('output_dim')}차원)" if reducer_meta else ""),
        f"- 군집화: {algorithm}",
        "- 대표 agent 선정: 군집 중심성과 쟁점 역할을 함께 고려",
        "- 대표 토론/보고서: LLM 또는 로컬 규칙 기반 fallback",
    ]


def build_report_markdown(
    policy_text: str,
    responses: list[CitizenResponse],
    representatives: list[RepresentativeAgent],
    debate: dict[str, Any],
    satisfaction: dict[str, Any],
    cluster_summary: list[dict[str, Any]],
    cluster_run_summary: dict[str, Any],
    warnings: list[str],
) -> str:
    lines = [
        f"# 정책 반응 보고서 - {policy_title(policy_text)}",
        "",
        f"- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 대표 agent 수: {len(representatives)}명",
        "",
        "## 만족도 분포 (토론 전 원본 여론)",
        "",
        "> 아래 분포는 토론에 반영되기 전 100명 합성 시민의 원본 만족도입니다. "
        "토론은 대표 시민만 참여하며, 이 분포를 재계산하지 않습니다(토론 후 여론 변화 수치가 아님).",
        "",
        "```text",
        satisfaction_block(satisfaction),
        "```",
        "",
        *format_technology_pipeline(cluster_run_summary),
        "",
        build_one_page_summary(policy_text, representatives, debate, responses, satisfaction),
    ]

    if cluster_summary:
        lines.extend(["", "## 군집 요약", ""])
        for cluster in cluster_summary:
            cluster_name = clean_space(cluster.get("cluster_name") or f"cluster {cluster.get('cluster_id', '')}")
            size = cluster.get("size", "")
            avg_score = cluster.get("avg_score", "")
            top_kw = cluster.get("top_keywords", [])
            if isinstance(top_kw, list):
                top_kw_text = ", ".join(
                    item[0] if isinstance(item, list) and item else str(item)
                    for item in top_kw[:5]
                )
            else:
                top_kw_text = clean_space(top_kw)
            lines.append(f"- {cluster_name}: {size}명, 평균 {avg_score}, 주요 키워드 {top_kw_text}")

    if warnings:
        lines.extend(["", "## 실행 참고", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(["", format_all_opinions(responses), "", format_debate_transcript(representatives, debate)])
    return "\n".join(lines) + "\n"


def build_transcript_text(
    policy_text: str,
    representatives: list[RepresentativeAgent],
    debate: dict[str, Any],
) -> str:
    return "\n".join(
        [
            f"정책: {policy_title(policy_text)}",
            "",
            format_debate_transcript(representatives, debate),
        ]
    )


# ---------------------------------------------------------------------------
# REPORT_SPEC.md 형식 1차 보고서
# 팀 렌더링 명세(REPORT_SPEC.md)에 맞춰 동적 항목만 시뮬레이션 결과에서 채우고,
# "○" 고정 문구는 정책과 무관하게 항상 동일하게 출력한다.
# 대표 시민 수는 절대 고정하지 않고 representative_citizens 배열 길이만큼 반복한다.
# ---------------------------------------------------------------------------

# 고정 문구(모든 정책 공통). 데이터에 따라 수정하지 않는다.
FIXED_PURPOSE_LINES = (
    "정책 시행 전 가상시민 100명의 반응을 측정·분석하여 정책 보완을 위한 기초자료로 활용하고자 함.",
    "본 보고서는 실제 주민 설문 결과가 아니라, 화순군 관련 데이터와 정책 시나리오를 바탕으로 생성한 시뮬레이션 결과임.",
)
FIXED_ANALYSIS_ITEMS = ("만족도", "가상시민 의견", "우려 키워드", "토론 쟁점", "개선 필요사항")
DEFAULT_SATISFACTION_RULE = (
    "각 가상시민은 정책 반응을 1~5점 리커트 척도로 응답함"
    "(1=매우 불만족, 2=불만족, 3=보통/조건부, 4=만족, 5=매우 만족). "
    "종합 만족도는 100명 응답의 산술평균을 5.0점 만점으로 나타낸 값임."
)
DEFAULT_SELECTION_RULE = (
    "가상시민 의견을 임베딩·군집화하여 유사 의견 집단을 나눈 뒤, "
    "각 군집의 중심성과 쟁점 대표성(직접수혜 여부·읍면 접근성·세대 차이 등 이해관계 대비)을 "
    "고려해 대표 토론자를 선정함."
)


def spec_stance(score: float, existing: str = "") -> str:
    """가상시민의 입장을 찬성/반대/중립 세 값으로 정규화한다."""
    existing = clean_space(existing)
    if "반대" in existing:
        return "반대"
    if "찬성" in existing:
        return "찬성"
    if existing in ("중립", "유보"):
        return "중립"
    if score >= 3.5:
        return "찬성"
    if score <= 2.5:
        return "반대"
    return "중립"


def overall_opinion_phrase(average: float) -> str:
    if average >= 4.0:
        return "전반적으로 긍정적이며 정책 도입에 대한 기대가 높은 편"
    if average >= 3.5:
        return "대체로 긍정적이나 일부 보완을 전제로 한 조건부 지지"
    if average >= 3.0:
        return "찬반이 갈리며 실행 조건에 따라 판단을 유보하는 신중한 기류"
    if average >= 2.5:
        return "우려가 다소 우세하여 실질적 보완이 필요하다는 의견이 다수"
    return "부정적 반응이 우세하여 정책 재설계 수준의 보완이 필요"


def spec_risk_factors(responses: list[CitizenResponse], limit: int = 3) -> list[str]:
    """만족도가 낮을수록 가중치를 높여, 주요 우려 요인(자원 쟁점)을 뽑는다."""
    counter: Counter[str] = Counter()
    for response in responses:
        if response.satisfaction_score <= 2:
            weight = 2.0
        elif response.satisfaction_score <= 3:
            weight = 1.5
        elif response.satisfaction_score >= 4:
            weight = 0.3
        else:
            weight = 1.0
        for tag in extract_resource_tags(response.opinion_text, response.summary_keyword):
            if tag != "기타":
                counter[tag] += weight
    factors = [tag for tag, _ in counter.most_common(limit)]
    if factors:
        return factors
    # 자원 태그가 없으면 저만족 시민의 키워드로 대체.
    fallback: Counter[str] = Counter()
    for response in responses:
        if response.satisfaction_score <= 3 and response.summary_keyword:
            fallback[response.summary_keyword] += 1
    return [keyword for keyword, _ in fallback.most_common(limit)] or ["확인된 주요 우려 요인 없음"]


def spec_citizen_keywords(response: CitizenResponse, limit: int = 3) -> list[str]:
    keywords: list[str] = []
    if response.summary_keyword:
        keywords.append(response.summary_keyword)
    for tag in extract_resource_tags(response.opinion_text, response.summary_keyword):
        if tag != "기타" and tag not in keywords:
            keywords.append(tag)
    return keywords[:limit] or ["키워드 없음"]


def stance_group_summary(
    responses: list[CitizenResponse],
    target_stance: str,
    limit: int = 6,
) -> dict[str, Any]:
    """찬성/반대/중립 그룹을 '한 줄 한 줄' 온전한 의견 목록으로 돌려준다(문장 자르지 않음)."""
    grouped = [
        response
        for response in responses
        if spec_stance(response.satisfaction_score, response.stance) == target_stance
    ]
    grouped.sort(
        key=lambda item: item.satisfaction_score,
        reverse=(target_stance == "찬성"),
    )
    items = [
        {
            "id": item.persona_id,
            "keyword": item.summary_keyword or "키워드 없음",
            "opinion": natural_polite(item.opinion_text),  # 자르지 않고 전체 문장
        }
        for item in grouped[:limit]
    ]
    return {"count": len(grouped), "shown": len(items), "items": items}


def summarize_scenario(policy: dict[str, Any], policy_summary: str, limit: int = 300) -> str:
    """정책 원문(main_details)이 매우 길 수 있으므로 핵심만 2~3문장으로 요약한다."""
    text = clean_space(policy_summary)
    # '정책개요'/'정책 목적' 라벨이 있으면 그 구간을 우선 사용한다.
    match = re.search(r"정책\s*개요\s*[:：]\s*(.+)", text)
    body = match.group(1) if match else text
    # 다음 라벨(세부 사업명·사업대상·신청 자격 등) 전까지만 자른다.
    body = re.split(
        r"(세부\s*사업명|사업\s*대상|모집\s*물량|신청\s*자격|지원\s*내용|접수\s*기간|선정\s*절차)\s*[:：]",
        body,
    )[0]
    sentences = [s.strip() for s in re.split(r"(?<=다[\.\s])\s*", body) if s.strip()]
    summary = " ".join(sentences[:3]).strip() or text
    # 본문을 먼저 자른 뒤 '(주요 대상: …)'을 통째로 덧붙인다.
    # (뒤에 자르면 괄호 안에서 잘려 '(주요 대상: … 취학 전…' 처럼 괄호가 안 닫힘)
    summary = shorten(summary, limit)
    target = clean_space(policy.get("target_audience"))
    if target and target not in summary:
        summary += f" (주요 대상: {shorten(target, 80)})"
    return summary


def dedupe_similar(points: list[str], limit: int = 4) -> list[str]:
    """의미가 겹치는 문장(예: '고령층 별도 지원' 반복)을 합쳐 대표 문장만 남긴다."""
    selected: list[str] = []
    for point in points:
        point = clean_space(point)
        if not point:
            continue
        if issues_overlap(point, selected):
            continue
        selected.append(point)
        if len(selected) >= limit:
            break
    return selected


def _speaker_label(agent: RepresentativeAgent | None, persona_id: str) -> str:
    """토론 전문에서 화자 이름 앞에 붙일 짧은 라벨(지역·나이·직업 요약)."""
    if agent is None:
        return persona_id
    bits: list[str] = []
    if agent.age is not None:
        bits.append(f"{agent.age}세")
    if agent.occupation:
        bits.append(agent.occupation)
    return f"{persona_id}({'·'.join(bits)})" if bits else persona_id


def build_debate_transcript(
    debate: dict[str, Any],
    representatives: list[RepresentativeAgent],
    policy_name: str,
) -> dict[str, Any]:
    """실제 토론 데이터(입론·반대신문·답변·재질문·조정·최종요구)를 사회자-패널 전문으로 엮는다."""
    lookup = {agent.persona_id: agent for agent in representatives}

    def speaker_of(pid: str) -> dict[str, str]:
        pid = clean_space(pid)
        agent = lookup.get(pid) or lookup.get(_canonical_pid(pid))
        return normalize_speaker(agent, pid)

    panelists = [
        {
            "id": agent.persona_id,
            "label": persona_label(agent),
            "name": normalize_speaker(agent, agent.persona_id)["name"],
            "detail": normalize_speaker(agent, agent.persona_id)["detail"],
            "stance": spec_stance(agent.satisfaction_score, agent.stance),
        }
        for agent in representatives
    ]

    turns: list[dict[str, str]] = []

    def moderator(text: str) -> None:
        turns.append({"speaker": "진행자", "speaker_detail": "", "role": "moderator", "text": clean_space(text)})

    def panel(pid: str, text: str) -> None:
        text = natural_polite(text)
        if text:
            who = speaker_of(pid)
            turns.append({"speaker": who["name"], "speaker_detail": who["detail"], "role": "panel", "text": text})

    # 1) 입론 — 입장을 먼저 밝히고 생활상의 이유를 이어 말한다.
    moderator(
        f"지금부터 '{policy_name}'에 대한 화순군 대표 시민 토론을 시작하겠습니다. "
        "먼저 각 대표께서 정책에 대한 입장과 그 이유를 말씀해 주시기 바랍니다."
    )
    for opening in debate.get("openings", []):
        pid = clean_space(opening.get("id"))
        position = clean_space(opening.get("입장") or opening.get("claim"))
        ground = clean_space(opening.get("핵심근거") or (opening.get("ground") or {}).get("내용"))
        position_text = format_position_statement(position)
        panel(pid, join_sentences([position_text, ground]))

    # 2) 반대신문(질문 → 답변 → 재질문/보완요구)
    questions = {item.get("pair_id"): item for item in debate.get("questions", [])}
    answers = {item.get("pair_id"): item for item in debate.get("answers", [])}
    followups = {item.get("pair_id"): item for item in debate.get("followups", [])}
    pair_ids = [pair.get("pair_id") for pair in debate.get("pairs", [])]
    if pair_ids:
        moderator("이제 대표들 사이의 반대신문을 진행하겠습니다. 서로 입장이 다른 지점을 짚어 질문해 주십시오.")
    for pair_id in pair_ids:
        question = questions.get(pair_id, {})
        answer = answers.get(pair_id, {})
        followup = followups.get(pair_id, {})
        asker = clean_space(question.get("from_id"))
        target = clean_space(question.get("to_id") or question.get("지목_대상"))
        panel(asker, question.get("질문"))
        panel(target, answer.get("답변"))
        panel(asker, join_sentences([clean_space(followup.get("재질문")), clean_space(followup.get("보완요구"))]))

    # 3) 쟁점별 조정
    mediations = debate.get("mediations", [])
    if mediations:
        moderator("쟁점별로 조정 토론을 이어가겠습니다.")
    for mediation in mediations:
        issue = clean_space(mediation.get("쟁점"))
        moderator(f"[{issue}] {clean_space(mediation.get('사회자질문'))}")
        for response in mediation.get("대표응답", []):
            pid = clean_space(response.get("id"))
            opinion = clean_space(response.get("조정의견"))
            concession = clean_space(response.get("양보조건"))
            # A2: 양보조건이 짧은 요약 라벨('고령층 지원 병행' 등)이면 발화에 붙이지 않는다.
            parts = [opinion]
            if _looks_like_sentence(concession):
                parts.append(concession)
            panel(pid, join_sentences(parts))

    # 4) 최종 요구 — A2: '최종입장' 요약 라벨은 발화에 넣지 않고 실제 요구/양보만 낸다.
    finals = debate.get("final_proposals", [])
    if finals:
        moderator("마지막으로 각 대표의 최종 입장과 핵심 요구를 듣겠습니다.")
    for proposal in finals:
        pid = clean_space(proposal.get("id"))
        parts: list[str] = []
        if clean_space(proposal.get("핵심요구")):
            parts.append(f"저는 정책에서 다음 내용을 검토해 주셨으면 합니다: {clean_space(proposal.get('핵심요구'))}.")
        if clean_space(proposal.get("양보가능지점")):
            parts.append(f"다만 다음 조건이라면 받아들일 수 있습니다: {clean_space(proposal.get('양보가능지점'))}.")
        panel(pid, join_sentences(parts))

    # 마무리 정리는 회의록의 '토론 정리/결론' 섹션에서 다루므로 여기서는 발언 전환만 한다.
    return {"moderator": "화순군 정책토론 사회자", "panelists": panelists, "turns": turns}


def build_deliberation(
    policy_name: str,
    scenario_summary: str,
    satisfaction: dict[str, Any],
    risks: list[str],
    issues: list[str],
    consensus: list[str],
    improvements: list[dict[str, str]],
    discussion_summary: str,
    dialogue: dict[str, Any],
) -> dict[str, Any]:
    """대표 시민 토론을 '공론화 회의록'(개요→배경→쟁점→주민 의견→토론 정리→권고안→결론)으로 엮는다.

    새 목소리(전문가·공무원)를 만들지 않고, 이미 생성된 대표 발언·쟁점·개선요구만
    사회자 시점에서 재구성한다.
    """
    average = float(satisfaction.get("average", 0.0))
    panel_count = len(dialogue.get("panelists", []))

    overview = (
        f"본 공론화는 '{policy_name}'에 대해 화순군 가상시민 대표들이 모여, "
        "찬반을 단순히 가르기보다 지역사회가 함께 수용할 수 있는 합리적 보완 기준을 찾기 위해 마련되었다."
    )
    risk_text = ", ".join(risks) if risks else "특이 우려 없음"
    background = (
        f"{scenario_summary} "
        f"가상시민 100명의 종합 만족도는 {average:.2f}점 / 5.0점으로 나타났으며, "
        f"주요 우려 요인은 {risk_text} 등으로 확인되었다. "
        "정책 취지에는 공감하나 직접 수혜 여부와 접근성 등에서 체감 격차가 제기되었다."
    )
    clash = ", ".join(issues[:2]) if issues else "핵심 쟁점"
    if consensus:
        agree_line = f"{consensus[0]} 같은 조건이 전제되면 일부 수용할 수 있다는 데 공감하였다."
    else:
        agree_line = "구체적 조건이 제시되면 판단을 조정할 수 있다는 여지를 남겼다."
    discussion = (
        f"토론을 통해 확인된 점은, 시행 여부 자체보다 '{discussion_summary}'에 대한 보완을 "
        "어떻게 설계하느냐가 정책 수용성을 좌우한다는 것이다. "
        f"대표 {panel_count}명은 {clash}에서 입장이 갈렸으나, {agree_line}"
    )
    top_reco = shorten(improvements[0]["title"], 40) if improvements else "제기된 보완 요구"
    conclusion = (
        f"이번 공론화는 '{policy_name}'을 둘러싼 반응이 단순 찬반이 아니라, "
        "정책의 필요성과 체감 격차 사이의 균형 문제임을 보여주었다. "
        "대표들은 소외되는 대상을 함께 챙기는 보완이 전제될 때 정책 수용성이 높아진다는 데 의견을 모았다. "
        f"화순군은 이번 결과를 바탕으로 '{top_reco}' 등 개선안을 검토할 필요가 있다."
    )
    return {
        "overview": overview,
        "background": background,
        "key_issues": issues,
        "dialogue": dialogue,
        "discussion_summary": discussion,
        "recommendations": improvements,
        "conclusion": conclusion,
    }


def representative_speech(agent: RepresentativeAgent, debate: dict[str, Any]) -> str:
    opening_by_id = {opening.get("id"): opening for opening in debate.get("openings", [])}
    final_by_id = {proposal.get("id"): proposal for proposal in debate.get("final_proposals", [])}
    opening = opening_by_id.get(agent.persona_id, {})
    final = final_by_id.get(agent.persona_id, {})
    parts: list[str] = []
    position = clean_space(opening.get("입장") or opening.get("claim"))
    if position:
        parts.append(f"저는 이 정책에 대해 '{position}'라는 입장입니다.")
    if agent.opinion_text:
        parts.append(clean_space(agent.opinion_text))
    demand = clean_space(final.get("핵심요구"))
    if demand:
        parts.append(f"최종적으로는 다음 내용을 검토해 주셨으면 합니다: {demand}.")
    concession = clean_space(final.get("양보가능지점"))
    if concession:
        parts.append(f"다만 다음 조건이라면 받아들일 수 있습니다: {concession}.")
    return natural_polite(" ".join(parts)) or UNVERIFIED_TEXT


def build_report_data(
    policy: dict[str, Any],
    policy_text: str,
    responses: list[CitizenResponse],
    representatives: list[RepresentativeAgent],
    debate: dict[str, Any],
    satisfaction: dict[str, Any],
) -> dict[str, Any]:
    """REPORT_SPEC.md 렌더링에 필요한 동적 데이터 묶음을 만든다."""
    average = float(satisfaction.get("average", 0.0))
    counts: dict[int, int] = {int(k): int(v) for k, v in satisfaction.get("counts", {}).items()}

    moderator = debate.get("moderator_summary") or {}

    # 대표 토론 요약: 찬성/반대/중립 + 주요 쟁점 + 합의(양보) 지점.
    # 주요 쟁점 — 조정 라운드의 쟁점 이름을 쓰되, 질문형/장문은 빼고 유사 쟁점은 합친다.
    raw_issues = [clean_space(item.get("쟁점")) for item in debate.get("mediations", [])]
    raw_issues += [clean_space(item) for item in moderator.get("핵심쟁점", [])]
    issues: list[str] = []
    for issue in raw_issues:
        if not issue or "?" in issue or len(issue) > 40:  # 질문형·장문 쟁점은 제외
            continue
        if issues_overlap(issue, issues):
            continue
        issues.append(issue)
        if len(issues) >= 5:
            break

    # 합의점 — 반복되는 유사 문장(예: '고령층 별도 지원')을 대표 문장으로 합친다.
    consensus_raw: list[str] = []
    for mediation in debate.get("mediations", []):
        for response in mediation.get("대표응답", []):
            consensus_raw.append(clean_space(response.get("양보조건")))
    if not any(consensus_raw):
        for proposal in debate.get("final_proposals", []):
            consensus_raw.append(clean_space(proposal.get("양보가능지점")))
    consensus_points = dedupe_similar([point for point in consensus_raw if point], limit=4)

    unresolved = [clean_space(item) for item in moderator.get("미해결쟁점", []) if clean_space(item)]
    discussion_summary = ", ".join(unresolved[:3]) or (", ".join(issues[:3]) or "핵심 쟁점")

    # 개선 필요사항: 최종요구를 중복·유사 제거해 title/reason으로 정리.
    improvements: list[dict[str, str]] = []
    selected_titles: list[str] = []
    final_lookup = {agent.persona_id: agent for agent in representatives}
    for proposal in debate.get("final_proposals", []):
        demand = clean_space(proposal.get("핵심요구"))
        if not demand or issues_overlap(demand, selected_titles):  # 유사 요구는 하나로
            continue
        selected_titles.append(demand)
        keyword = clean_space(proposal.get("키워드")) or "정책 보완"
        speaker = clean_space(proposal.get("id"))
        agent = final_lookup.get(speaker)
        speaker_label = persona_label(agent) if agent else speaker
        concession = clean_space(proposal.get("양보가능지점"))
        reason = f"'{keyword}' 쟁점에서 {speaker_label} 대표가 제기한 보완 요구."
        if concession:
            reason += f" 양보 가능 지점: {concession}"
        improvements.append({"title": demand, "reason": reason})

    # 대표 토론 요약 — 찬반 집계가 아니라 '대표자'만, 각자의 대표 키워드로 보여준다.
    opening_by_id = {opening.get("id"): opening for opening in debate.get("openings", [])}
    representative_summary = [
        {
            "id": agent.persona_id,
            "label": persona_label(agent),
            "keyword": agent.summary_keyword or "키워드 없음",
            "stance": spec_stance(agent.satisfaction_score, agent.stance),
            "position": clean_space(
                (opening_by_id.get(agent.persona_id, {}).get("입장"))
                or (opening_by_id.get(agent.persona_id, {}).get("claim"))
            )
            or score_position(agent.satisfaction_score, agent.summary_keyword),
            "opinion": shorten(agent.opinion_text, 100),
        }
        for agent in representatives
    ]

    # 응답자 구성 설명(지역·연령 분포 기반).
    region_counter = Counter(response.region_name_kr for response in responses if response.region_name_kr)
    ages = [response.age for response in responses if response.age is not None]
    region_text = ", ".join(f"{name} {count}명" for name, count in region_counter.most_common(5))
    age_text = (
        f"평균 {sum(ages) / len(ages):.0f}세(최소 {min(ages)}세~최대 {max(ages)}세)"
        if ages
        else "연령 정보 미상"
    )
    respondent_description = (
        "화순군 읍·면별 인구 분포와 연령·성별·직업 구성을 반영해 생성한 가상시민 100명으로 구성됨. "
        + (f"주요 응답 지역: {region_text}. " if region_text else "")
        + f"연령 구성: {age_text}."
    )

    report_policy_name = clean_space(policy.get("policy_name") or policy.get("name")) or policy_title(policy_text)
    scenario_summary = summarize_scenario(policy, clean_space(policy.get("main_details")) or policy_text)
    risks = spec_risk_factors(responses, limit=3)
    dialogue = build_debate_transcript(debate, representatives, report_policy_name)
    deliberation = build_deliberation(
        policy_name=report_policy_name,
        scenario_summary=scenario_summary,
        satisfaction=satisfaction,
        risks=risks,
        issues=issues,
        consensus=consensus_points,
        improvements=improvements,
        discussion_summary=discussion_summary,
        dialogue=dialogue,
    )

    overall_opinion = overall_opinion_phrase(average)
    data: dict[str, Any] = {
        "policy_name": report_policy_name,
        "overall_score": round(average, 2),
        "policy_summary": clean_space(policy.get("main_details"))
        or clean_space(policy.get("policy_name"))
        or policy_text,
        "scenario_summary": scenario_summary,
        "respondent_description": respondent_description,
        "satisfaction_rule": DEFAULT_SATISFACTION_RULE,
        "selection_rule": DEFAULT_SELECTION_RULE,
        "overall_opinion": overall_opinion,
        # A3: '{overall_opinion}으로' 조합 시 조사 오류(필요으로/기류으로)를 미리 교정한 형태
        "overall_opinion_ro": clean_text(f"{overall_opinion}으로"),
        "risks": risks,
        "discussion_summary": discussion_summary,
        "rating_distribution": [{"score": score, "count": counts.get(score, 0)} for score in range(1, 6)],
        "top_keywords": [keyword for keyword, _ in top_keywords(responses, limit=5)],
        "representative_summary": representative_summary,
        "support_summary": stance_group_summary(responses, "찬성"),
        "opposition_summary": stance_group_summary(responses, "반대"),
        "neutral_summary": stance_group_summary(responses, "중립"),
        "issues": issues or ["토론 기록상 뚜렷한 쟁점이 확인되지 않음"],
        "consensus": consensus_points or ["합의보다는 조건부 수용 지점만 확인됨"],
        "improvements": improvements,
        "representative_citizens": [
            {
                "id": agent.persona_id,
                "label": persona_label(agent),
                "stance": spec_stance(agent.satisfaction_score, agent.stance),
                "speech": representative_speech(agent, debate),
            }
            for agent in representatives
        ],
        "citizen_results": [
            {
                "id": response.persona_id,
                "score": int(round(response.satisfaction_score)),
                "keywords": spec_citizen_keywords(response),
                "opinion": shorten(natural_polite(response.opinion_text), 120),
            }
            for response in sorted(responses, key=_citizen_sort_key)
        ],
        "debate_transcript": dialogue,
        "deliberation": deliberation,
    }

    # 렌더 직전 후처리: 모든 생성 텍스트의 조합 흔적을 정리하고 검출 로그를 남긴다.
    cleaner_log: list[str] = []
    data = deep_clean(data, cleaner_log)
    data["_cleaner_log"] = cleaner_log
    return data


def _citizen_sort_key(response: CitizenResponse) -> tuple[str, str]:
    match = re.search(r"(\d+)", response.persona_id)
    number = f"{int(match.group(1)):06d}" if match else response.persona_id
    return number, response.persona_id


def _rating_bar_chart(rating_distribution: list[dict[str, int]]) -> str:
    total = sum(item["count"] for item in rating_distribution) or 1
    lines: list[str] = []
    for item in rating_distribution:
        score = int(item["score"])
        count = int(item["count"])
        ratio = count / total * 100
        bar = "█" * count if count else ""
        lines.append(f"{score}점 {SCORE_LABELS.get(score, ''):<7} | {bar:<50} {count:3d}명 ({ratio:5.1f}%)")
    return "\n".join(lines)


def _stance_lines_md(group: Any) -> list[str]:
    """찬성/반대/중립 그룹을 '한 줄 한 줄' 온전한 의견 목록(마크다운)으로 만든다."""
    if not isinstance(group, dict) or not group.get("items"):
        return ["_해당 입장의 의견은 두드러지지 않음._"]
    lines = [f"가상시민 **{group['count']}명**이 이 입장으로 분류됨 (대표 의견 {group['shown']}건):", ""]
    for item in group["items"]:
        lines.append(f"- **{item['id']}** ({item['keyword']}) — {item['opinion']}")
    remaining = group["count"] - group["shown"]
    if remaining > 0:
        lines.append(f"- … 외 {remaining}명")
    return lines


def _dialogue_md(dialogue: dict[str, Any]) -> list[str]:
    """주민 대표 발언을 진행자>/패널> 형식 마크다운으로 만든다."""
    lines: list[str] = [f"▶ **진행** : {dialogue.get('moderator', '사회자')}", "", "▶ **패널** :"]
    for panelist in dialogue.get("panelists", []):
        lines.append(f"　{panelist['label']} — {panelist['stance']}")
    lines += [""]
    for turn in dialogue.get("turns", []):
        lines.append(f"**{turn['speaker']}>** {turn['text']}")
        lines.append("")
    return lines


def _deliberation_md(deliberation: dict[str, Any]) -> list[str]:
    """대표 시민 토론을 공론화 회의록 형식 마크다운으로 만든다."""
    lines: list[str] = [
        "## 1. 회의 개요",
        "",
        f"○ {deliberation['overview']}",
        "",
        "## 2. 논의 배경",
        "",
        deliberation["background"],
        "",
        "## 3. 주요 쟁점",
        "",
    ]
    lines += [f"- {issue}" for issue in deliberation["key_issues"]]
    lines += ["", "## 4. 주민(대표) 의견", ""]
    lines += _dialogue_md(deliberation["dialogue"])
    lines += [
        "## 5. 토론 정리",
        "",
        f"○ {deliberation['discussion_summary']}",
        "",
        "## 6. 권고안",
        "",
    ]
    if deliberation["recommendations"]:
        for index, item in enumerate(deliberation["recommendations"], start=1):
            lines.append(f"{index}. **{item['title']}** — {item['reason']}")
    else:
        lines.append("도출된 권고안이 없음.")
    lines += [
        "",
        "## 7. 결론",
        "",
        f"○ {deliberation['conclusion']}",
    ]
    return lines


def render_spec_report(data: dict[str, Any]) -> str:
    """build_report_data 결과를 REPORT_SPEC.md 형식 마크다운으로 렌더링한다."""
    policy_name = data["policy_name"]
    overall_score = f"{data['overall_score']:.1f}"
    lines: list[str] = []

    # 표지
    lines += [
        "# 화순군",
        "",
        f"# {policy_name} 가상시민 반응 시뮬레이션 결과",
        "## 1차 보고서",
        "",
        "---",
        "",
    ]

    # 개요
    lines += [
        "# 개요",
        "",
        f"- **정책명** : {policy_name}",
        "- **분석단위** : 화순군 가상시민 100명 반응",
        "- **분석내용** :",
    ]
    lines += [f"    - {item}" for item in ("만족도", "가상시민 100명 의견", "우려 키워드", "토론 쟁점", "개선 필요사항")]
    lines += [
        "",
        f"- **종합 만족도** : {overall_score} / 5.0점",
        "",
        "---",
        "",
    ]

    # Ⅰ. 보고 목적 및 분석 개요
    lines += [
        "# Ⅰ. 보고 목적 및 분석 개요",
        "",
        "## 1. 보고 목적",
        "",
    ]
    lines += [f"○ {line}" for line in FIXED_PURPOSE_LINES]
    lines += [
        "",
        "## 2. 분석 개요",
        "",
        f"- **분석 대상** : {policy_name}",
        "- **가상 시민** : 화순군 인구·지역 특성을 반영한 가상시민 100명",
        "- **출력 항목** :",
    ]
    lines += [f"    - {item}" for item in FIXED_ANALYSIS_ITEMS]
    lines += ["", "---", ""]

    # Ⅱ. 정책 시나리오 (원문이 길어 핵심만 요약)
    lines += [
        "# Ⅱ. 정책 시나리오",
        "",
        data.get("scenario_summary") or data["policy_summary"],
        "",
        "---",
        "",
    ]

    # Ⅲ. 구성 및 평가 기준
    lines += [
        "# Ⅲ. 가상시민 100명 구성 및 평가 기준",
        "",
        "## 1. 응답자 구성 기준",
        "",
        data["respondent_description"],
        "",
        "## 2. 만족도 산정 기준",
        "",
        data["satisfaction_rule"],
        "",
        "## 3. 대표 토론자 선정 원칙",
        "",
        data["selection_rule"],
        "",
        "---",
        "",
    ]

    # Ⅳ. 조사분석
    risks = data["risks"]
    lines += [
        "# Ⅳ. 조사분석 : 만족도 및 토론 요약",
        "",
        "## 1. 총평",
        "",
        f"○ 가상시민 100명 반응 분석 결과, 종합 만족도는 {overall_score}점 / 5.0점으로 나타남.",
        "",
        f"○ 정책에 대한 전반적 반응은 {data['overall_opinion_ro']} 분석되었으며, 주요 우려 요인은 다음과 같이 나타남.",
        "",
    ]
    lines += [f"- {risk}" for risk in risks]
    lines += [
        "",
        f"○ 대표 시민 토론 결과, 정책의 필요성에는 일정 부분 공감하였으나 "
        f"{data['discussion_summary']}에 대한 보완이 필요한 것으로 분석됨.",
        "",
        "## 2. 만족도 분포",
        "",
        "```text",
        _rating_bar_chart(data["rating_distribution"]),
        "```",
        "",
        "## 3. 주요 키워드 TOP5",
        "",
    ]
    for rank, keyword in enumerate(data["top_keywords"], start=1):
        lines.append(f"{rank}. {keyword}")
    lines += ["", "## 4. 대표 토론 요약", "", "○ 토론에 참여한 대표별 핵심 키워드와 입장은 다음과 같음.", ""]
    for rep in data.get("representative_summary", []):
        lines.append(
            f"- **{rep['keyword']}** · {rep['label']} ({rep['stance']}) — {rep['opinion']}"
        )
    lines += ["", "### 주요 쟁점", ""]
    lines += [f"- {issue}" for issue in data["issues"]]
    lines += ["", "### 합의점", ""]
    lines += [f"- {point}" for point in data["consensus"]]
    lines += ["", "---", ""]

    # Ⅴ. 개선 필요사항
    lines += [
        "# Ⅴ. 개선 필요사항",
        "",
        "○ 가상시민 100명 의견 및 대표 시민 토론 결과를 종합한 개선 필요사항은 다음과 같음.",
        "",
    ]
    if data["improvements"]:
        for index, improvement in enumerate(data["improvements"], start=1):
            lines += [
                f"### {index}.",
                "",
                f"- **개선 필요사항** : {improvement['title']}",
                f"- **근거** : {improvement['reason']}",
                "",
            ]
    else:
        lines += ["도출된 개선 필요사항이 없음.", ""]
    lines += ["---", ""]

    # 부록 1. 가상시민 100명 의견 전문
    lines += [
        "# 부록 1. 가상시민 100명 의견 전문",
        "",
        "○ 아래 표는 가상시민별 만족도, 주요 우려 키워드, 1문장 의견 결과임.",
        "",
        "| ID | 점수 | 키워드 | 1문장 의견 |",
        "| --- | --- | --- | --- |",
    ]
    for citizen in data["citizen_results"]:
        keyword_text = ", ".join(citizen["keywords"])
        opinion_text = citizen["opinion"].replace("|", "/")
        lines.append(f"| {citizen['id']} | {citizen['score']} | {keyword_text} | {opinion_text} |")
    lines += ["", "---", ""]

    # 부록 2. 대표 시민 공론화 회의록 (대표 시민 수 고정하지 않음)
    lines += ["# 부록 2. 대표 시민 공론화 회의록", ""]
    lines += _deliberation_md(data.get("deliberation", {}))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# REPORT_SPEC.md 1차 보고서 — 브라우저용 HTML 렌더링
# 별도 라이브러리 없이 인라인 CSS만 사용해, 파일을 더블클릭하면 브라우저에서
# 만족도 막대그래프·키워드·찬반 카드·표가 보기 좋게 표시되도록 만든다.
# ---------------------------------------------------------------------------
_SCORE_COLORS = {
    1: "#e5484d",  # 매우 불만족 - 빨강
    2: "#f2820c",  # 불만족 - 주황
    3: "#8b8d98",  # 보통 - 회색
    4: "#46a758",  # 만족 - 연녹
    5: "#2a7e3b",  # 매우 만족 - 진녹
}
_STANCE_COLORS = {"찬성": "#2a7e3b", "반대": "#e5484d", "중립": "#8b8d98"}


def _esc(text: Any) -> str:
    import html

    return html.escape(str(text if text is not None else ""))


def _rating_chart_html(rating_distribution: list[dict[str, int]]) -> str:
    # [B1] 막대 width를 '전체 100명 대비 실제 백분율'로 계산해 옆 라벨(%)과 일치시킨다.
    total = sum(item["count"] for item in rating_distribution) or 1
    rows: list[str] = []
    for item in rating_distribution:
        score = int(item["score"])
        count = int(item["count"])
        ratio = count / total * 100  # 라벨과 동일한 값이 곧 막대 width
        color = _SCORE_COLORS.get(score, "#8b8d98")
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{score}점 <span>{_esc(SCORE_LABELS.get(score, ""))}</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{ratio:.1f}%;background:{color}"></div></div>'
            f'<div class="bar-value">{count}명 <span>({ratio:.1f}%)</span></div>'
            f"</div>"
        )
    return '<div class="bar-chart">' + "".join(rows) + "</div>"


def _keywords_html(top_keywords: list[str]) -> str:
    chips: list[str] = []
    for rank, keyword in enumerate(top_keywords):
        size = 1.55 - rank * 0.16          # 순위 강조는 글자 크기로만
        opacity = max(0.7, 1.0 - rank * 0.1)  # [B3] opacity 하한 0.7 — 연초록 배경에서 가독성 확보
        chips.append(
            f'<span class="kw" style="font-size:{size:.2f}rem;opacity:{opacity:.2f}">'
            f'<b>{rank + 1}</b> {_esc(keyword)}</span>'
        )
    return '<div class="kw-cloud">' + "".join(chips) + "</div>"


def _stance_card(title: str, group: Any, color: str) -> str:
    """찬성/반대/중립 카드 — 의견을 한 줄 한 줄(문장 자르지 않고) 목록으로 표시."""
    if not isinstance(group, dict) or not group.get("items"):
        body = '<p class="muted">해당 입장의 의견은 두드러지지 않음.</p>'
        count_badge = ""
    else:
        rows = "".join(
            f'<li><b>{_esc(item["id"])}</b> <span class="kw-tag">{_esc(item["keyword"])}</span>'
            f'<br>{_esc(item["opinion"])}</li>'
            for item in group["items"]
        )
        remaining = group["count"] - group["shown"]
        more = f'<li class="muted">… 외 {remaining}명</li>' if remaining > 0 else ""
        body = f'<ul class="stance-list">{rows}{more}</ul>'
        count_badge = f'<span class="stance-count" style="background:{color}">{group["count"]}명</span>'
    return (
        f'<div class="stance-card" style="border-top:4px solid {color}">'
        f'<h4 style="color:{color}">{_esc(title)} {count_badge}</h4>'
        f"{body}</div>"
    )


def _dialogue_html(dialogue: dict[str, Any]) -> str:
    """주민 대표 발언을 진행자/패널 대화 형식 HTML로 렌더링."""
    panel_intro = "".join(
        f'<div class="panelist"><span class="stance-dot" style="background:{_STANCE_COLORS.get(p["stance"], "#8b8d98")}"></span>'
        f'<b>{_esc(p.get("name") or p.get("label"))}</b> '
        f'<span class="muted">— {_esc(p["stance"])}'
        + (f' · {_esc(p["detail"])}' if p.get("detail") else "")
        + "</span></div>"
        for p in dialogue.get("panelists", [])
    )
    # [B4] 화자명은 짧게(HW016 · 30세), 직업/지역 상세는 title 툴팁 + 작은 부제로.
    def _turn_html(turn: dict[str, str]) -> str:
        cls = "turn moderator" if turn.get("role") == "moderator" else "turn"
        detail = clean_space(turn.get("speaker_detail"))
        who = (
            f'<span class="who" title="{_esc(detail)}">{_esc(turn["speaker"])}'
            + (f'<span class="who-sub">{_esc(detail)}</span>' if detail else "")
            + "</span>"
        )
        return f'<div class="{cls}">{who}<span class="say">{_esc(turn["text"])}</span></div>'

    turns = "".join(_turn_html(turn) for turn in dialogue.get("turns", []))
    return (
        f'<div class="panel-box"><div class="panel-title">▶ 진행 : {_esc(dialogue.get("moderator", "사회자"))}</div>'
        f'<div class="panel-title">▶ 패널</div>{panel_intro}</div>'
        f'<div class="transcript">{turns}</div>'
    )


def _deliberation_html(deliberation: dict[str, Any]) -> str:
    """대표 시민 토론을 공론화 회의록 형식 HTML로 렌더링."""
    issues = "".join(f"<li>{_esc(issue)}</li>" for issue in deliberation.get("key_issues", []))
    if deliberation.get("recommendations"):
        recos = "".join(
            f'<div class="improve-item"><div class="improve-no">{i}</div>'
            f'<div class="improve-body"><div class="improve-title">{_esc(item["title"])}</div>'
            f'<div class="improve-reason">{_esc(item["reason"])}</div></div></div>'
            for i, item in enumerate(deliberation["recommendations"], start=1)
        )
    else:
        recos = '<p class="muted">도출된 권고안이 없음.</p>'
    return (
        f'<h3>1. 회의 개요</h3><div class="fixed">○ {_esc(deliberation.get("overview"))}</div>'
        f'<h3>2. 논의 배경</h3><p>{_esc(deliberation.get("background"))}</p>'
        f'<h3>3. 주요 쟁점</h3><ul>{issues}</ul>'
        f'<h3>4. 주민(대표) 의견</h3>{_dialogue_html(deliberation.get("dialogue", {}))}'
        f'<h3>5. 토론 정리</h3><div class="fixed">○ {_esc(deliberation.get("discussion_summary"))}</div>'
        f'<h3>6. 권고안</h3>{recos}'
        f'<h3>7. 결론</h3><div class="fixed">○ {_esc(deliberation.get("conclusion"))}</div>'
    )


def render_spec_report_html(data: dict[str, Any]) -> str:
    """build_report_data 결과를 브라우저에서 보기 좋은 HTML 보고서로 렌더링한다."""
    policy_name = _esc(data["policy_name"])
    overall = float(data["overall_score"])
    # [B2] 1~5 리커트 게이지: (score-1)/4 로 채워 최저점이 0%에 가깝게(만점만 100%).
    overall_pct = max(0.0, min(100.0, (overall - 1) / 4 * 100))

    # 우려 요인 / 쟁점 / 합의점 / 개선사항 리스트 조립
    risks_html = "".join(f"<li>{_esc(risk)}</li>" for risk in data["risks"])
    issues_html = "".join(f"<li>{_esc(issue)}</li>" for issue in data["issues"])
    consensus_html = "".join(f"<li>{_esc(point)}</li>" for point in data["consensus"])

    # 대표 토론 요약 — 대표자만, 대표 키워드 기준(찬반 집계 아님)
    reps_summary_html = "".join(
        f'<div class="rep-summary"><span class="rep-kw">{_esc(rep["keyword"])}</span>'
        f'<span class="stance-badge" style="background:{_STANCE_COLORS.get(rep["stance"], "#8b8d98")}">{_esc(rep["stance"])}</span>'
        f'<div class="rep-summary-body"><b>{_esc(rep["label"])}</b><br>{_esc(rep["opinion"])}</div></div>'
        for rep in data.get("representative_summary", [])
    )

    if data["improvements"]:
        improvements_html = "".join(
            f'<div class="improve-item"><div class="improve-no">{index}</div>'
            f'<div class="improve-body"><div class="improve-title">{_esc(item["title"])}</div>'
            f'<div class="improve-reason">{_esc(item["reason"])}</div></div></div>'
            for index, item in enumerate(data["improvements"], start=1)
        )
    else:
        improvements_html = '<p class="muted">도출된 개선 필요사항이 없음.</p>'

    # 부록 1 — 100명 표. [B6] 키워드는 쉼표 텍스트 대신 .kw-tag pill로 렌더(기존 CSS 재활용).
    def _kw_cell(keywords: list[str]) -> str:
        return "".join(f'<span class="kw-tag">{_esc(k)}</span>' for k in keywords)

    citizen_rows = "".join(
        f"<tr><td class='cid'>{_esc(c['id'])}</td>"
        f"<td class='score-cell'><span class='score-pill' style='background:{_SCORE_COLORS.get(int(c['score']), '#8b8d98')}'>{_esc(c['score'])}</span></td>"
        f"<td class='kw-cell'>{_kw_cell(c['keywords'])}</td>"
        f"<td class='opinion'>{_esc(c['opinion'])}</td></tr>"
        for c in data["citizen_results"]
    )

    # 부록 2 — 실제 토론 전문(진행자/패널). 대표 시민 수는 배열 길이만큼.
    rep_html = _deliberation_html(data.get("deliberation", {}))

    analysis_items = "".join(
        f"<li>{_esc(item)}</li>"
        for item in ("만족도", "가상시민 의견", "우려 키워드", "토론 쟁점", "개선 필요사항")
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{policy_name} 가상시민 반응 시뮬레이션 1차 보고서</title>
<style>
  :root {{ --green:#2a7e3b; --green-d:#1f5f2c; --ink:#1a1d1f; --muted:#6b7280; --line:#e5e7eb; --bg:#f5f6f7; --card:#ffffff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:'Pretendard','Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR',sans-serif;
    line-height:1.7; -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:0 20px 80px; }}
  .cover {{ background:linear-gradient(135deg,var(--green),var(--green-d)); color:#fff;
    padding:56px 40px; border-radius:0 0 24px 24px; text-align:center; box-shadow:0 8px 24px rgba(0,0,0,.12); }}
  .cover .badge {{ display:inline-block; background:rgba(255,255,255,.18); padding:6px 16px; border-radius:999px;
    font-size:.85rem; letter-spacing:.05em; margin-bottom:18px; }}
  .cover h1 {{ margin:.2em 0; font-size:1.9rem; font-weight:800; }}
  .cover .sub {{ font-size:1.05rem; opacity:.92; }}
  section {{ background:var(--card); border:1px solid var(--line); border-radius:16px;
    padding:28px 32px; margin-top:22px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
  h2 {{ font-size:1.35rem; margin:0 0 18px; padding-bottom:12px; border-bottom:2px solid var(--green);
    color:var(--green-d); display:flex; align-items:center; gap:10px; }}
  h3 {{ font-size:1.08rem; margin:22px 0 10px; color:#111; }}
  .fixed {{ background:#f0f7f1; border-left:4px solid var(--green); padding:12px 16px; border-radius:8px;
    margin:10px 0; color:#28502f; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .meta-row {{ display:flex; gap:12px; padding:8px 0; border-bottom:1px dashed var(--line); }}
  .meta-row .k {{ min-width:110px; color:var(--muted); font-weight:600; }}
  ul {{ margin:8px 0; padding-left:20px; }}
  li {{ margin:4px 0; }}
  .muted {{ color:var(--muted); }}
  /* 종합 만족도 배너 */
  .score-banner {{ display:flex; align-items:center; gap:28px; background:linear-gradient(135deg,#f0f7f1,#e4f0e6);
    border-radius:14px; padding:24px 28px; }}
  .score-big {{ font-size:3.4rem; font-weight:800; color:var(--green-d); line-height:1; }}
  .score-big small {{ font-size:1.1rem; color:var(--muted); font-weight:600; }}
  .score-meter {{ flex:1; }}
  .score-meter .track {{ height:14px; background:#d9e5db; border-radius:999px; overflow:hidden; }}
  .score-meter .fill {{ height:100%; background:linear-gradient(90deg,var(--green),var(--green-d)); border-radius:999px; }}
  /* 만족도 막대 */
  .bar-chart {{ display:flex; flex-direction:column; gap:9px; margin-top:8px; }}
  .bar-row {{ display:flex; align-items:center; gap:12px; }}
  .bar-label {{ min-width:120px; font-weight:600; font-size:.92rem; }}
  .bar-label span {{ color:var(--muted); font-weight:400; }}
  .bar-track {{ flex:1; background:#eef0f2; border-radius:8px; height:22px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:8px; transition:width .4s; }}
  .bar-value {{ min-width:96px; text-align:right; font-variant-numeric:tabular-nums; font-size:.9rem; }}
  .bar-value span {{ color:var(--muted); }}
  /* 키워드 */
  .kw-cloud {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:8px 0; }}
  .kw {{ background:#eef6ef; color:var(--green-d); padding:8px 16px; border-radius:999px; font-weight:600; }}
  .kw b {{ color:var(--green); margin-right:4px; }}
  /* 찬반 카드 */
  .stance-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
  .stance-card {{ background:#fafbfc; border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .stance-card h4 {{ margin:0 0 8px; font-size:1rem; display:flex; align-items:center; gap:8px; }}
  .stance-card p {{ margin:0; font-size:.9rem; color:#374151; }}
  .stance-count {{ color:#fff; font-size:.72rem; padding:2px 9px; border-radius:999px; font-weight:700; }}
  .stance-list {{ margin:0; padding-left:18px; font-size:.88rem; color:#374151; }}
  .stance-list li {{ margin:8px 0; line-height:1.55; }}
  .kw-tag {{ display:inline-block; background:#eef6ef; color:var(--green-d); padding:1px 8px; border-radius:999px;
    font-size:.75rem; font-weight:600; margin-left:4px; }}
  /* 대표 토론 요약(대표자별) */
  .rep-summary-list {{ display:flex; flex-direction:column; gap:10px; }}
  .rep-summary {{ display:grid; grid-template-columns:auto auto 1fr; align-items:center; gap:12px;
    background:#fafbfc; border:1px solid var(--line); border-radius:12px; padding:12px 16px; }}
  .rep-kw {{ background:var(--green-d); color:#fff; font-weight:700; padding:4px 12px; border-radius:8px; font-size:.85rem; white-space:nowrap; }}
  .rep-summary-body {{ font-size:.9rem; color:#374151; }}
  /* 개선사항 */
  .improve-item {{ display:flex; gap:16px; padding:16px; background:#fafbfc; border:1px solid var(--line);
    border-radius:12px; margin-bottom:12px; }}
  .improve-no {{ flex:none; width:34px; height:34px; border-radius:50%; background:var(--green); color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700; }}
  .improve-title {{ font-weight:700; margin-bottom:4px; }}
  .improve-reason {{ font-size:.9rem; color:var(--muted); }}
  /* 표 */
  .table-scroll {{ max-height:520px; overflow:auto; border:1px solid var(--line); border-radius:12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  thead th {{ position:sticky; top:0; background:var(--green-d); color:#fff; padding:10px 12px; text-align:left; z-index:1; }}
  td {{ padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
  tbody tr:nth-child(even) {{ background:#fafbfc; }}
  .cid {{ font-weight:600; white-space:nowrap; }}
  .score-cell {{ text-align:center; }}
  .score-pill {{ display:inline-block; min-width:26px; padding:2px 8px; border-radius:999px; color:#fff; font-weight:700; }}
  .opinion {{ color:#374151; }}
  /* 토론 전문(진행자/패널) */
  .panel-box {{ background:#f0f7f1; border:1px solid #d7e7da; border-radius:12px; padding:16px 20px; margin-bottom:20px; }}
  .panel-title {{ font-weight:800; color:var(--green-d); margin:6px 0 4px; }}
  .panelist {{ display:flex; align-items:center; gap:8px; padding:3px 0 3px 6px; font-size:.92rem; }}
  .stance-dot {{ flex:none; width:10px; height:10px; border-radius:50%; }}
  .transcript {{ display:flex; flex-direction:column; gap:14px; }}
  .turn {{ display:flex; gap:12px; }}
  .turn .who {{ flex:none; min-width:120px; max-width:120px; font-weight:700; color:#111; display:flex; flex-direction:column; gap:2px; }}
  .who-sub {{ font-size:.72rem; font-weight:400; color:var(--muted); line-height:1.2; word-break:keep-all; }}
  .kw-cell {{ display:flex; flex-wrap:wrap; gap:4px; }}
  .turn .say {{ flex:1; background:#fafbfc; border:1px solid var(--line); border-radius:0 12px 12px 12px;
    padding:10px 14px; color:#374151; }}
  .turn.moderator .who {{ color:var(--green-d); }}
  .turn.moderator .say {{ background:#eef6ef; border-color:#d7e7da; font-weight:500; }}
  footer {{ text-align:center; color:var(--muted); font-size:.85rem; margin-top:32px; }}
  @media (max-width:680px) {{ .grid2,.stance-grid {{ grid-template-columns:1fr; }} .score-banner {{ flex-direction:column; text-align:center; }}
    .turn {{ flex-direction:column; gap:4px; }} .turn .who {{ max-width:none; }} }}
</style>
</head>
<body>
<div class="cover">
  <div class="badge">화순군 · 가상시민 반응 시뮬레이션</div>
  <h1>{policy_name} 가상시민 반응 시뮬레이션 결과</h1>
  <div class="sub">1차 보고서</div>
</div>
<div class="wrap">

  <section>
    <h2>개요</h2>
    <div class="meta-row"><span class="k">정책명</span><span>{policy_name}</span></div>
    <div class="meta-row"><span class="k">분석단위</span><span>화순군 가상시민 100명 반응</span></div>
    <div class="meta-row"><span class="k">분석내용</span><span>만족도 · 가상시민 100명 의견 · 우려 키워드 · 토론 쟁점 · 개선 필요사항</span></div>
    <h3>종합 만족도</h3>
    <div class="score-banner">
      <div class="score-big">{overall:.1f}<small> / 5.0점</small></div>
      <div class="score-meter">
        <div class="track"><div class="fill" style="width:{overall_pct:.1f}%"></div></div>
        <p class="muted" style="margin:8px 0 0">{_esc(data["overall_opinion"])}</p>
      </div>
    </div>
  </section>

  <section>
    <h2>Ⅰ. 보고 목적 및 분석 개요</h2>
    <h3>1. 보고 목적</h3>
    <div class="fixed">○ 정책 시행 전 가상시민 100명의 반응을 측정·분석하여 정책 보완을 위한 기초자료로 활용하고자 함.</div>
    <div class="fixed">○ 본 보고서는 실제 주민 설문 결과가 아니라, 화순군 관련 데이터와 정책 시나리오를 바탕으로 생성한 시뮬레이션 결과임.</div>
    <h3>2. 분석 개요</h3>
    <div class="grid2">
      <div><div class="meta-row"><span class="k">분석 대상</span><span>{policy_name}</span></div>
        <div class="meta-row"><span class="k">가상 시민</span><span>화순군 인구·지역 특성을 반영한 가상시민 100명</span></div></div>
      <div><b>출력 항목</b><ul>{analysis_items}</ul></div>
    </div>
  </section>

  <section>
    <h2>Ⅱ. 정책 시나리오</h2>
    <p>{_esc(data.get("scenario_summary") or data["policy_summary"])}</p>
  </section>

  <section>
    <h2>Ⅲ. 가상시민 100명 구성 및 평가 기준</h2>
    <h3>1. 응답자 구성 기준</h3><p>{_esc(data["respondent_description"])}</p>
    <h3>2. 만족도 산정 기준</h3><p>{_esc(data["satisfaction_rule"])}</p>
    <h3>3. 대표 토론자 선정 원칙</h3><p>{_esc(data["selection_rule"])}</p>
  </section>

  <section>
    <h2>Ⅳ. 조사분석 : 만족도 및 토론 요약</h2>
    <h3>1. 총평</h3>
    <div class="fixed">○ 가상시민 100명 반응 분석 결과, 종합 만족도는 {overall:.1f}점 / 5.0점으로 나타남.</div>
    <div class="fixed">○ 정책에 대한 전반적 반응은 <b>{_esc(data["overall_opinion_ro"])}</b> 분석되었으며, 주요 우려 요인은 다음과 같음.</div>
    <ul>{risks_html}</ul>
    <div class="fixed">○ 대표 시민 토론 결과, 정책의 필요성에는 일정 부분 공감하였으나 <b>{_esc(data["discussion_summary"])}</b>에 대한 보완이 필요한 것으로 분석됨.</div>
    <h3>2. 만족도 분포</h3>
    {_rating_chart_html(data["rating_distribution"])}
    <h3>3. 주요 키워드 TOP5</h3>
    {_keywords_html(data["top_keywords"])}
    <h3>4. 대표 토론 요약</h3>
    <p class="muted">토론에 참여한 대표별 핵심 키워드와 입장입니다.</p>
    <div class="rep-summary-list">{reps_summary_html}</div>
    <div class="grid2" style="margin-top:16px">
      <div><h3 style="margin-top:0">주요 쟁점</h3><ul>{issues_html}</ul></div>
      <div><h3 style="margin-top:0">합의점</h3><ul>{consensus_html}</ul></div>
    </div>
  </section>

  <section>
    <h2>Ⅴ. 개선 필요사항</h2>
    <div class="fixed">○ 가상시민 100명 의견 및 대표 시민 토론 결과를 종합한 개선 필요사항은 다음과 같음.</div>
    {improvements_html}
  </section>

  <section>
    <h2>부록 1. 가상시민 100명 의견 전문</h2>
    <p class="muted">가상시민별 만족도, 주요 우려 키워드, 1문장 의견 결과입니다. (표 안에서 스크롤됩니다)</p>
    <div class="table-scroll">
      <table><thead><tr><th>ID</th><th>점수</th><th>키워드</th><th>1문장 의견</th></tr></thead>
      <tbody>{citizen_rows}</tbody></table>
    </div>
  </section>

  <section>
    <h2>부록 2. 대표 시민 공론화 회의록</h2>
    {rep_html}
  </section>

  <footer>화순군 가상시민 반응 시뮬레이션 · 1차 보고서 · 자동 생성 (본 결과는 시뮬레이션이며 실제 설문이 아님)</footer>
</div>
</body>
</html>
"""


def maybe_write_docx(markdown_text: str, path: Path) -> bool:
    try:
        from docx import Document
    except ModuleNotFoundError:
        return False

    document = Document()
    in_code_block = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not line:
            document.add_paragraph("")
            continue
        if in_code_block:
            document.add_paragraph(line)
        elif line.startswith("# "):
            document.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            document.add_heading(line[3:], level=2)
        elif line.startswith("### "):
            document.add_heading(line[4:], level=3)
        elif line.startswith("- "):
            document.add_paragraph(line[2:], style="List Bullet")
        else:
            document.add_paragraph(line)
    document.save(path)
    return True


def create_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root).expanduser().resolve() / f"hwasun_debate_report_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    input_files: dict[str, str],
    policy: dict[str, Any],
    policy_text: str,
    responses: list[CitizenResponse],
    representatives: list[RepresentativeAgent],
    cluster_summary: list[dict[str, Any]],
    cluster_run_summary: dict[str, Any],
    debate: dict[str, Any],
    satisfaction: dict[str, Any],
    warnings: list[str],
) -> dict[str, str]:
    report_md = build_report_markdown(
        policy_text=policy_text,
        responses=responses,
        representatives=representatives,
        debate=debate,
        satisfaction=satisfaction,
        cluster_summary=cluster_summary,
        cluster_run_summary=cluster_run_summary,
        warnings=warnings,
    )
    report_path = output_dir / "policy_reaction_report.md"
    transcript_path = output_dir / "debate_transcript.txt"
    result_path = output_dir / "debate_result.json"
    report_path.write_text(report_md, encoding=UTF8_SIG)
    transcript_path.write_text(
        build_transcript_text(policy_text, representatives, debate),
        encoding=UTF8_SIG,
    )

    # REPORT_SPEC.md 형식 1차 보고서 + 렌더링용 데이터 묶음.
    report_data = build_report_data(
        policy=policy,
        policy_text=policy_text,
        responses=responses,
        representatives=representatives,
        debate=debate,
        satisfaction=satisfaction,
    )
    # 렌더 직전 텍스트 클리너가 검출·수정한 항목을 실행 로그로 남긴다.
    for entry in report_data.get("_cleaner_log", []):
        warnings.append(entry)
    spec_report_path = output_dir / "policy_report_1st.md"
    spec_html_path = output_dir / "policy_report_1st.html"
    report_data_path = output_dir / "report_data.json"
    spec_report_path.write_text(render_spec_report(report_data), encoding=UTF8_SIG)
    spec_html_path.write_text(render_spec_report_html(report_data), encoding="utf-8")
    write_json(report_data_path, report_data)

    result_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_files": input_files,
        "policy": policy,
        "policy_text": policy_text,
        "satisfaction": satisfaction,
        "representatives": [asdict(agent) for agent in representatives],
        "cluster_summary": cluster_summary,
        "cluster_run_summary": cluster_run_summary,
        "debate": debate,
        "warnings": warnings,
        "report_files": {
            "markdown": str(report_path),
            "spec_report": str(spec_report_path),
            "spec_report_html": str(spec_html_path),
            "report_data": str(report_data_path),
            "transcript": str(transcript_path),
        },
    }
    write_json(result_path, result_payload)

    output_files = {
        "json": str(result_path),
        "markdown": str(report_path),
        "spec_report": str(spec_report_path),
        "spec_report_html": str(spec_html_path),
        "report_data": str(report_data_path),
        "transcript": str(transcript_path),
    }
    if args.docx:
        docx_path = output_dir / "policy_reaction_report.docx"
        if maybe_write_docx(report_md, docx_path):
            output_files["docx"] = str(docx_path)
        else:
            warnings.append("python-docx가 없어 DOCX 보고서는 건너뛰었습니다.")
            write_json(result_path, {**result_payload, "warnings": warnings})
    return output_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="화순군 대표 agent 토론 및 정책 반응 보고서 생성")
    parser.add_argument("--responses", default=str(DEFAULT_RESPONSE_PATH), help="100명 정책 반응 JSON/CSV 경로")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="outputs 폴더 경로")
    parser.add_argument("--cluster-dir", default="", help="군집화 결과 폴더. representative_agents.json을 포함해야 함")
    parser.add_argument("--representatives", default="", help="representative_agents.json 직접 경로")
    parser.add_argument("--output-dir", default="", help="토론/보고서 출력 폴더 직접 지정")
    parser.add_argument("--rag-db", default=str(DEFAULT_RAG_DB_PATH), help="증거팩 생성을 위한 Town-Twin SQLite DB 경로")
    parser.add_argument("--max-cross-exams", type=int, default=3, help="반대신문으로 뽑을 최대 쌍 수")
    parser.add_argument(
        "--api-mode",
        choices=("auto", "api", "local"),
        default="auto",
        help="auto: 가능하면 API, 안 되면 로컬 / api: API 필수 / local: 로컬 규칙 기반",
    )
    parser.add_argument("--no-api", action="store_true", help="--api-mode local과 동일")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI 토론 모델 (기본값: {DEFAULT_MODEL})",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--docx", action="store_true", help="가능하면 DOCX 보고서도 생성")
    parser.add_argument(
        "--no-ensure-beneficiary",
        action="store_true",
        help="직접 수혜·찬성 대표 자동 보강을 끄고 군집화가 뽑은 대표만 사용",
    )
    parser.add_argument(
        "--dedupe-representatives",
        action="store_true",
        help="같은 주장을 반복하는 유사 대표를 명시적으로 통합",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warnings: list[str] = []

    response_path = Path(args.responses).expanduser().resolve()
    representative_path = resolve_representative_path(args)
    cluster_dir = representative_path.parent
    output_dir = create_output_dir(args)

    policy, responses, _response_meta = load_policy_responses(response_path)
    representatives = load_representatives(representative_path, responses)
    # 기본값은 군집 대표를 모두 유지한다. 명시적으로 요청한 경우에만 유사 대표를 통합한다.
    if args.dedupe_representatives:
        representatives = dedupe_representatives(representatives, warnings)
    # 직접 수혜·찬성 당사자가 대표에 반드시 1명은 포함되도록 보강한다.
    if not args.no_ensure_beneficiary:
        representatives = ensure_direct_beneficiary_representative(representatives, responses, warnings)
    cluster_summary = load_cluster_summary(cluster_dir)
    cluster_run_summary = load_cluster_run_summary(cluster_dir)
    policy_text = policy_to_text(policy, responses)
    add_policy_tags_to_agents(representatives, policy_text)
    evidence_packs = build_evidence_packs(
        representatives,
        policy_text,
        Path(args.rag_db).expanduser().resolve(),
        warnings,
    )

    satisfaction = satisfaction_summary(responses)
    client = build_debate_client(args, warnings)
    debate = run_debate(
        agents=representatives,
        policy_text=policy_text,
        client=client,
        max_cross_exams=max(0, args.max_cross_exams),
        evidence_packs=evidence_packs,
        warnings=warnings,
    )

    input_files = {
        "responses": str(response_path),
        "representatives": str(representative_path),
        "cluster_dir": str(cluster_dir),
        "rag_db": str(Path(args.rag_db).expanduser().resolve()),
    }
    output_files = write_outputs(
        output_dir=output_dir,
        args=args,
        input_files=input_files,
        policy=policy,
        policy_text=policy_text,
        responses=responses,
        representatives=representatives,
        cluster_summary=cluster_summary,
        cluster_run_summary=cluster_run_summary,
        debate=debate,
        satisfaction=satisfaction,
        warnings=warnings,
    )

    if not args.quiet:
        print("토론 및 정책 반응 보고서 생성 완료")
        print(f"- 대표 agent: {len(representatives)}명")
        print(f"- 평균 만족도: {satisfaction['average']:.2f}/5.0")
        print(f"- 출력 폴더: {output_dir}")
        for label, path in output_files.items():
            print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
