#!/usr/bin/env python3
"""after_100.py — 정책 수정 후(2차) 재시뮬레이션 및 전/후 비교 보고서.

general_100.py와 동일한 매커니즘으로 동작한다.

    수정된 정책안 입력 → 동일 가상시민 100명 의견 생성 → 만족도 분포·평균 확인

달라지는 점은, 1차(general_100) 결과와 비교하여 변화를 확인할 수 있는
2차 비교 보고서(REPORT_SPEC_V2.md 형식)를 output으로 내보낸다는 것이다.

전제 파이프라인:
    1차) general_100.py       → hwasun_persona_policy_responses.json (수정 전 100명 결과)
         토론.py             → report_data.json (1차 보고서: 개선 필요사항·대표 시민 등)
    2차) after_100.py         → 수정 정책 재시뮬레이션 + 전/후 비교 보고서

기본 실행(수정 정책을 넣고 재시뮬레이션 후 비교):
    cd /home/piai/다운로드/POBICA
    python after_100.py \
     --policy-file hwasun_24h_childcare_policy_after.txt \
     --before hwasun_persona_policy_responses.json \
    --previous-report outputs/hwasun_debate_report_20260712_002529/report_data.json

이미 돌려둔 2차 결과로 보고서만 다시 만들기(재시뮬레이션 생략):
    python after_100.py --after-file hwasun_persona_policy_responses_after.json \
        --before hwasun_persona_policy_responses.json \
        --previous-report outputs/<report>/report_data.json

출력:
    after_outputs/hwasun_persona_policy_responses_after.json / .csv
    after_outputs/policy_report_2nd.md
    after_outputs/policy_report_2nd.html
    after_outputs/report_data_2nd.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# general_100.py의 시뮬레이션 엔진을 그대로 재사용한다(같은 폴더에 있어야 함).
from general_100 import (
    DEFAULT_DB_PATH,
    DEFAULT_PERSONA_PATH,
    HwasunPersonaPolicySimulator,
    PolicyInputParser,
    UTF8_SIG,
    load_env_file,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REVISED_POLICY_PATH = SCRIPT_DIR / "hwasun_100won_hyodo_taxi_policy_2nd.txt"
DEFAULT_BEFORE_PATH = SCRIPT_DIR / "hwasun_persona_policy_responses_d3.json"
DEFAULT_PREVIOUS_REPORT = (
    SCRIPT_DIR / "outputs" / "hwasun_debate_report_20260712_213551" / "report_data.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "after_outputs"
DEFAULT_AFTER_OUTPUT = DEFAULT_OUTPUT_DIR / "hwasun_persona_policy_responses_after.json"
DEFAULT_REPORT_OUTPUT = DEFAULT_OUTPUT_DIR / "policy_report_2nd.md"
DEFAULT_HTML_OUTPUT = DEFAULT_OUTPUT_DIR / "policy_report_2nd.html"
DEFAULT_DATA_OUTPUT = DEFAULT_OUTPUT_DIR / "report_data_2nd.json"

SCORE_LABELS = {
    1: "매우 불만족",
    2: "불만족",
    3: "보통/조건부",
    4: "만족",
    5: "매우 만족",
}

# 고정 문구(모든 정책 공통). "○"가 붙은 문장은 데이터에 따라 수정하지 않는다.
FIXED_PURPOSE_LINES = (
    "본 보고서는 1차 시뮬레이션에서 도출된 개선 필요사항을 바탕으로 정책을 수정한 뒤, "
    "동일한 가상시민 100명을 대상으로 다시 시뮬레이션하여 정책 수정 효과를 비교·분석하고자 함.",
    "본 보고서는 실제 주민 설문 결과가 아니라, 화순군 관련 데이터와 수정 정책 시나리오를 "
    "바탕으로 생성한 시뮬레이션 결과임.",
)
FIXED_ANALYSIS_OVERVIEW = (
    "수정 전(1차)과 수정 후(2차)를 동일한 가상시민 100명 기준으로 1:1 비교함.",
    "단순 만족도 증감뿐 아니라 새롭게 해결된 문제, 여전히 남은 문제, 새로 발생한 문제, "
    "만족도 변화 원인을 함께 분석함.",
)
FIXED_REEVAL_TARGET = (
    "○ 재평가 대상은 1차와 동일한 가상시민 100명이며, 페르소나 구성(지역·연령·직업 등)은 "
    "변경하지 않고 정책 시나리오만 수정하여 반응을 재측정함."
)
FIXED_REFLECTION_RULE = (
    "○ 개선사항 반영 기준은 1차 보고서의 개선 필요사항이 수정 정책에 반영된 정도로 판정하며, "
    "완전 반영(Resolved, 1.0점) / 부분 반영(Partially Resolved, 0.5점) / 미반영(Unresolved, 0점) "
    "3단계로 구분함. 반영률(CRR)은 (점수 총합 ÷ 개선사항 개수) × 100 으로 자동 계산함."
)

# 개선사항 반영 판정에서 키워드 매칭 시 잡음이 되는 일반 어휘.
_STOPWORDS = {
    "함께", "효과", "높임", "방식", "설계", "수용", "지점", "대상", "관련", "보완",
    "요구", "기준", "가능", "정책", "운영", "지역", "시민", "경우", "부터", "우선",
    "단계", "확대", "제공", "고려", "조정", "마련", "공개", "구체", "실제", "적용",
    "동시", "병행", "필요", "여건", "개선", "방문", "동선", "규모", "본관", "채널",
    "그대로", "정도",
}


# ---------------------------------------------------------------------------
# 입출력 유틸
# ---------------------------------------------------------------------------
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding=UTF8_SIG)


def clean_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def shorten(text: str, limit: int = 120) -> str:
    text = clean_space(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


_POLITE_ENDINGS = (
    (r"괜찮겠더라구요", "괜찮겠더라고요"),
    (r"편하겠구먼요?", "편할 것 같아요"),
    (r"좋겠구먼요?", "좋을 것 같아요"),
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
    """의견 내용은 유지하고 반말·음슴체 종결만 자연스러운 해요체로 정리한다."""
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


# 만족도 점수별 색(분포 막대) — 토론.py와 동일한 의미색 유지.
_SCORE_COLORS = {1: "#e5484d", 2: "#f2820c", 3: "#8b8d98", 4: "#46a758", 5: "#2a7e3b"}


def summarize_revised_scenario(
    revised_policy: dict[str, Any], revised_policy_text: str, limit: int = 300
) -> str:
    """수정 정책 원문에서 핵심 개요만 2~3문장으로 요약한다(토론.py 시나리오 요약과 동일 방식).

    긴 원문(main_details)을 그대로 싣지 않고, '정책개요' 구간을 우선 잡아 세부 라벨
    전까지만 추린 뒤 앞 2~3문장 + 주요 대상만 남긴다. 괄호는 통째로만 덧붙여 미종결을 막는다.
    """
    text = clean_space(
        revised_policy.get("main_details") or revised_policy.get("details") or revised_policy_text
    )
    match = re.search(r"정책\s*개요\s*[:：]\s*(.+)", text)
    body = match.group(1) if match else text
    body = re.split(
        r"(세부\s*사업명|사업\s*대상|모집\s*물량|신청\s*자격|지원\s*내용|접수\s*기간|선정\s*절차)\s*[:：]",
        body,
    )[0]
    sentences = [s.strip() for s in re.split(r"(?<=다[\.\s])\s*", body) if s.strip()]
    summary = " ".join(sentences[:3]).strip() or text
    summary = shorten(summary, limit)
    target = clean_space(revised_policy.get("target_audience"))
    if target and target not in summary:
        summary += f" (주요 대상: {shorten(target, 80)})"
    return summary


def read_revised_policy_text(args: argparse.Namespace) -> str:
    if args.policy_text:
        return args.policy_text
    if args.policy_file:
        return read_text_auto(Path(args.policy_file))
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    try:
        text = input("수정된 정책을 입력하시오: ").strip()
    except EOFError as exc:
        raise ValueError(
            "수정 정책 입력이 필요합니다. --policy-text 또는 --policy-file, "
            "혹은 이미 돌려둔 결과라면 --after-file을 사용하세요."
        ) from exc
    if not text:
        raise ValueError("수정 정책 입력이 비어 있습니다.")
    return text


# ---------------------------------------------------------------------------
# 시뮬레이션 결과 표준화(general_100 payload 형식)
# ---------------------------------------------------------------------------
def load_simulation_payload(path: Path) -> dict[str, Any]:
    """general_100.py가 저장한 {policy, summary, results} 형식을 읽는다."""
    payload = load_json(path)
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError(f"시뮬레이션 결과 형식이 아닙니다(results 키 없음): {path}")
    return payload


def result_index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("persona_id")): row for row in results if row.get("persona_id")}


def clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 3
    return min(5, max(1, score))


def score_distribution(results: list[dict[str, Any]]) -> dict[int, int]:
    counts = {score: 0 for score in range(1, 6)}
    for row in results:
        counts[clamp_score(row.get("satisfaction_score"))] += 1
    return counts


def average_score(results: list[dict[str, Any]]) -> float:
    scores = [clamp_score(row.get("satisfaction_score")) for row in results]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def split_keywords(raw: Any) -> list[str]:
    parts = re.split(r"[,/·|]", str(raw or ""))
    return [part.strip() for part in parts if part.strip()]


def keyword_counter(results: list[dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for row in results:
        for keyword in split_keywords(row.get("summary_keyword")):
            counter[keyword] += 1
    return counter


def persona_sort_key(persona_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", persona_id)
    return (int(match.group(1)) if match else 10**9, persona_id)


def key_terms(text: str, limit: int = 12) -> list[str]:
    tokens = [
        token
        for token in re.split(r"[^0-9A-Za-z가-힣]+", str(text or ""))
        if len(token) >= 2 and token not in _STOPWORDS
    ]
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return seen[:limit]


# ---------------------------------------------------------------------------
# 전/후 비교 계산
# ---------------------------------------------------------------------------
def build_citizen_comparison(
    before_idx: dict[str, dict[str, Any]],
    after_idx: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    common_ids = sorted(set(before_idx) & set(after_idx), key=persona_sort_key)
    rows: list[dict[str, Any]] = []
    for pid in common_ids:
        before = before_idx[pid]
        after = after_idx[pid]
        before_score = clamp_score(before.get("satisfaction_score"))
        after_score = clamp_score(after.get("satisfaction_score"))
        rows.append(
            {
                "id": pid,
                "before_score": before_score,
                "after_score": after_score,
                "delta": after_score - before_score,
                "before_keywords": split_keywords(before.get("summary_keyword")),
                "after_keywords": split_keywords(after.get("summary_keyword")),
                "before_opinion": clean_space(before.get("opinion_text")),
                "after_opinion": clean_space(after.get("opinion_text")),
                "before_stance": clean_space(before.get("stance")),
                "after_stance": clean_space(after.get("stance")),
            }
        )
    return rows


def low_score_keywords(rows: list[dict[str, Any]], which: str) -> Counter:
    """저만족(<=2) 시민이 언급한 우려 키워드 분포."""
    counter: Counter = Counter()
    for row in rows:
        if row[f"{which}_score"] <= 2:
            for keyword in row[f"{which}_keywords"]:
                counter[keyword] += 1
    return counter


def build_opinion_change_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improved = [row for row in rows if row["delta"] > 0]
    worsened = [row for row in rows if row["delta"] < 0]

    positive_kw = Counter()
    for row in improved:
        for keyword in row["after_keywords"]:
            positive_kw[keyword] += 1

    negative_kw = Counter()
    for row in worsened:
        for keyword in row["after_keywords"]:
            negative_kw[keyword] += 1

    before_concern = low_score_keywords(rows, "before")
    after_concern = low_score_keywords(rows, "after")
    new_concerns = [kw for kw in after_concern if kw not in before_concern]
    resolved_concerns = [kw for kw in before_concern if kw not in after_concern]

    return {
        "improved_count": len(improved),
        "worsened_count": len(worsened),
        "unchanged_count": len(rows) - len(improved) - len(worsened),
        # 긍정 증가 요인: 만족도가 오른 시민들이 새로 짚은 초점.
        "positive_drivers": [kw for kw, _ in positive_kw.most_common(5)],
        # 부정 감소 요인: 저만족 상태에서 사라진 우려.
        "negative_reducers": resolved_concerns[:5],
        # 새롭게 등장한 우려 / 사라진 우려.
        "new_concerns": new_concerns[:5],
        "resolved_concerns": resolved_concerns[:5],
        # 만족도가 내려간 시민들이 짚은 초점(새 문제 신호).
        "worsening_drivers": [kw for kw, _ in negative_kw.most_common(5)],
    }


def load_improvements(
    previous_report_path: Path | None,
    before_payload: dict[str, Any],
) -> list[dict[str, str]]:
    """1차 보고서(report_data.json)의 개선 필요사항 배열을 읽는다."""
    for source in (previous_report_path, None):
        data: dict[str, Any] | None = None
        if source is not None and source.exists():
            candidate = load_json(source)
            if isinstance(candidate, dict):
                data = candidate
        elif source is None and isinstance(before_payload, dict):
            # before가 report_data.json 형식이면 거기서 improvements를 찾는다.
            data = before_payload if "improvements" in before_payload else None
        if not data:
            continue
        raw = data.get("improvements")
        if isinstance(raw, list) and raw:
            improvements: list[dict[str, str]] = []
            for item in raw:
                if isinstance(item, dict):
                    improvements.append(
                        {
                            "title": clean_space(item.get("title") or item.get("개선 필요사항")),
                            "reason": clean_space(item.get("reason") or item.get("근거")),
                        }
                    )
                elif item:
                    improvements.append({"title": clean_space(item), "reason": ""})
            return [imp for imp in improvements if imp["title"]]
    return []


def judge_reflection(improvement: dict[str, str], policy_text: str) -> dict[str, Any]:
    """개선 필요사항이 수정 정책에 반영됐는지 키워드 매칭으로 판정한다."""
    terms = key_terms(improvement["title"])
    if not terms:
        return {
            "verdict": "Unresolved",
            "score": 0.0,
            "matched": [],
            "reason": "개선 필요사항에서 판정할 핵심어를 추출하지 못함.",
        }
    policy_norm = clean_space(policy_text)
    matched = [term for term in terms if term in policy_norm]
    ratio = len(matched) / len(terms)
    if ratio >= 0.5:
        verdict, score = "Resolved", 1.0
    elif ratio >= 0.2:
        verdict, score = "Partially Resolved", 0.5
    else:
        verdict, score = "Unresolved", 0.0
    if matched:
        reason = (
            f"수정 정책에서 확인된 관련어: {', '.join(matched)} "
            f"(핵심어 {len(matched)}/{len(terms)} 일치)."
        )
    else:
        reason = "수정 정책에서 개선 필요사항 핵심어가 확인되지 않음."
    return {"verdict": verdict, "score": score, "matched": matched, "reason": reason}


def build_reflection_analysis(
    improvements: list[dict[str, str]],
    revised_policy_text: str,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for improvement in improvements:
        judged = judge_reflection(improvement, revised_policy_text)
        items.append({**improvement, **judged})
    total = len(items)
    total_score = sum(item["score"] for item in items)
    crr = round(total_score / total * 100, 1) if total else 0.0
    return {
        "items": items,
        "total": total,
        "resolved": sum(1 for item in items if item["verdict"] == "Resolved"),
        "partially": sum(1 for item in items if item["verdict"] == "Partially Resolved"),
        "unresolved": sum(1 for item in items if item["verdict"] == "Unresolved"),
        "score_sum": round(total_score, 1),
        "crr": crr,
    }


VERDICT_KR = {
    "Resolved": "완전 반영",
    "Partially Resolved": "부분 반영",
    "Unresolved": "미반영",
}


def overall_reaction_phrase(average: float) -> str:
    if average >= 4.0:
        return "전반적으로 긍정적이며 정책 도입에 대한 기대가 높은 편"
    if average >= 3.5:
        return "대체로 긍정적이나 일부 보완을 전제로 한 조건부 지지"
    if average >= 3.0:
        return "찬반이 갈리며 실행 조건에 따라 판단을 유보하는 신중한 기류"
    if average >= 2.5:
        return "우려가 다소 우세하여 추가 보완이 필요하다는 의견이 다수"
    return "부정적 반응이 우세하여 정책 재설계 수준의 보완이 필요"


def delta_phrase(delta: float) -> str:
    if delta > 0.3:
        return f"수정 전 대비 평균 {delta:+.2f}점 상승하여 뚜렷한 개선이 확인됨"
    if delta > 0.05:
        return f"수정 전 대비 평균 {delta:+.2f}점 소폭 상승함"
    if delta < -0.3:
        return f"수정 전 대비 평균 {delta:+.2f}점 하락하여 역효과가 우려됨"
    if delta < -0.05:
        return f"수정 전 대비 평균 {delta:+.2f}점 소폭 하락함"
    return "수정 전과 평균 만족도 차이가 뚜렷하지 않음"


def paired_ttest(citizen_rows: list[dict[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    """동일 시민 100명의 수정 전/후 만족도에 대해 대응표본(paired) t-검정을 수행한다.

    같은 페르소나를 전/후로 반복 측정한 구조이므로 대응표본 t-검정이 적합하다.
    귀무가설 H0: 수정 전과 후의 평균 만족도가 같다(차이 = 0).
    scipy가 있으면 scipy.stats.ttest_rel을 쓰고, 없으면 대표본 정규근사로 대체한다.
    """
    before = [row["before_score"] for row in citizen_rows]
    after = [row["after_score"] for row in citizen_rows]
    n = len(citizen_rows)
    diffs = [a - b for a, b in zip(after, before)]
    mean_before = round(sum(before) / n, 3) if n else 0.0
    mean_after = round(sum(after) / n, 3) if n else 0.0
    mean_diff = round(mean_after - mean_before, 3)

    result = {
        "n": n,
        "df": max(n - 1, 0),
        "alpha": alpha,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_diff": mean_diff,
        "t_stat": 0.0,
        "p_value": 1.0,
        "significant": False,
        "direction": "no_change",
        "method": "paired t-test",
    }

    # 표본이 2개 미만이거나 모든 차이가 동일(분산 0)하면 검정 불가.
    if n < 2 or all(d == diffs[0] for d in diffs):
        if mean_diff > 0:
            result["direction"] = "increase"
        elif mean_diff < 0:
            result["direction"] = "decrease"
        result["reason"] = "표본이 부족하거나 전/후 차이의 분산이 0이라 t-검정을 수행할 수 없음."
        return result

    try:
        from scipy import stats  # type: ignore

        t_stat, p_value = stats.ttest_rel(after, before)
        result["method"] = "paired t-test (scipy.stats.ttest_rel)"
    except Exception:
        import math

        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var_d / n) if var_d > 0 else 0.0
        t_stat = mean_d / se if se > 0 else 0.0
        # 대표본 정규근사: p ≈ 2 * (1 - Φ(|t|)).
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
        result["method"] = "paired t-test (정규근사)"

    t_stat = float(t_stat)
    p_value = float(p_value)
    significant = p_value < alpha
    if mean_diff > 0:
        direction = "increase"
    elif mean_diff < 0:
        direction = "decrease"
    else:
        direction = "no_change"

    result.update(
        {
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": bool(significant),
            "direction": direction,
        }
    )
    return result


def ttest_phrase(tt: dict[str, Any]) -> str:
    """t-검정 결과를 한 줄 문구로 요약한다."""
    if tt["significant"] and tt["direction"] == "increase":
        return "유의미한 만족도 상승을 보였습니다!"
    if tt["significant"] and tt["direction"] == "decrease":
        return "통계적으로 유의미한 만족도 하락이 확인되었습니다."
    return "수정 전후 만족도 차이는 통계적으로 유의미하지 않았습니다."


def ttest_detail(tt: dict[str, Any]) -> str:
    """t-검정 통계치를 한 줄 근거 문장으로 만든다."""
    sig = "유의미함" if tt["significant"] else "유의미하지 않음"
    return (
        f"대응표본 t-검정 결과 t({tt['df']}) = {tt['t_stat']:.3f}, "
        f"p = {tt['p_value']:.4f} (유의수준 α = {tt['alpha']}) → {sig}. "
        f"평균 만족도 {tt['mean_before']:.2f} → {tt['mean_after']:.2f} "
        f"(차이 {tt['mean_diff']:+.2f}점, n = {tt['n']}명)."
    )


def build_analysis(
    revised_policy: dict[str, Any],
    revised_policy_text: str,
    before_results: list[dict[str, Any]],
    after_results: list[dict[str, Any]],
    citizen_rows: list[dict[str, Any]],
    reflection: dict[str, Any],
    ttest: dict[str, Any],
) -> dict[str, Any]:
    before_avg = average_score(before_results)
    after_avg = average_score(after_results)
    delta = round(after_avg - before_avg, 3)

    opinion_change = build_opinion_change_analysis(citizen_rows)
    after_low_kw = low_score_keywords(citizen_rows, "after")

    biggest_improved = opinion_change["positive_drivers"][0] if opinion_change["positive_drivers"] else "뚜렷한 개선 초점 없음"
    still_lacking = (after_low_kw.most_common(1)[0][0] if after_low_kw else "추가 우려는 크게 확인되지 않음")

    policy_name = clean_space(revised_policy.get("policy_name") or revised_policy.get("name")) or "정책명 미상"
    # 수정된 내용 중심으로 요약(원문 통짜 대신 핵심 2~3문장 + 주요 대상).
    revised_summary = summarize_revised_scenario(revised_policy, revised_policy_text)
    # 이번 수정에 반영된 개선사항(1차 요구 중 반영으로 판정된 항목) — '수정된 내용'의 핵심.
    reflected_improvements = [
        clean_space(item.get("title"))
        for item in reflection.get("items", [])
        if item.get("verdict") in ("Resolved", "Partially Resolved")
    ]

    # 총평(Ⅳ.1) — 명세가 요구하는 5개 요소를 모두 포함.
    review_lines = [
        f"수정 후 종합 만족도는 {after_avg:.2f}점 / 5.0점으로 나타남.",
        f"{delta_phrase(delta)}(수정 전 {before_avg:.2f}점 → 수정 후 {after_avg:.2f}점).",
        f"전반적 반응은 {overall_reaction_phrase(after_avg)}으로 분석됨.",
        f"가장 크게 개선된 요소는 '{biggest_improved}' 관련 반응이며, "
        f"만족도가 오른 시민이 {opinion_change['improved_count']}명으로 확인됨.",
        f"아직 부족한 요소는 '{still_lacking}' 관련 우려이며, "
        f"수정 후에도 저만족(2점 이하) 시민이 {sum(1 for r in citizen_rows if r['after_score'] <= 2)}명 남아 있음.",
        ttest_detail(ttest),
    ]
    # 유의미한 상승이면 명세가 요구한 강조 문구를 총평 맨 앞에 추가.
    if ttest["significant"] and ttest["direction"] == "increase":
        review_lines.insert(0, ttest_phrase(ttest))

    return {
        "policy_name": policy_name,
        "revised_policy_summary": revised_summary,
        "before_score": before_avg,
        "after_score": after_avg,
        "delta_score": delta,
        "reflection_rate": reflection["crr"],
        "ttest": ttest,
        "ttest_phrase": ttest_phrase(ttest),
        "ttest_detail": ttest_detail(ttest),
        "review_lines": review_lines,
        "overall_reaction": overall_reaction_phrase(after_avg),
        "opinion_change": opinion_change,
        "biggest_improved": biggest_improved,
        "still_lacking": still_lacking,
        "revised_policy_summary_short": shorten(revised_summary, 400),
        "reflected_improvements": reflected_improvements,
    }


# ---------------------------------------------------------------------------
# 렌더링(REPORT_SPEC_V2.md)
# ---------------------------------------------------------------------------
def distribution_bar_chart(distribution: dict[int, int]) -> str:
    total = sum(distribution.values()) or 1
    lines: list[str] = []
    for score in range(1, 6):
        count = distribution.get(score, 0)
        ratio = count / total * 100
        bar = "█" * count if count else ""
        lines.append(f"{score}점 {SCORE_LABELS[score]:<7} | {bar:<50} {count:3d}명 ({ratio:5.1f}%)")
    return "\n".join(lines)


def comparison_bar_chart(before: dict[int, int], after: dict[int, int]) -> str:
    total_b = sum(before.values()) or 1
    total_a = sum(after.values()) or 1
    lines = [f"{'점수':<11} | {'수정 전(1차)':<22} | {'수정 후(2차)':<22} | 변화량"]
    lines.append("-" * 78)
    for score in range(1, 6):
        b = before.get(score, 0)
        a = after.get(score, 0)
        bar_b = "█" * round(b / total_b * 20)
        bar_a = "█" * round(a / total_a * 20)
        lines.append(
            f"{score}점 {SCORE_LABELS[score]:<7} | {bar_b:<16}{b:3d}명 | {bar_a:<16}{a:3d}명 | {a - b:+d}명"
        )
    return "\n".join(lines)


def mean_bar_chart(before_score: float, after_score: float) -> str:
    def bar(value: float) -> str:
        return "█" * round(value / 5.0 * 40)

    return "\n".join(
        [
            f"수정 전(1차) | {bar(before_score):<40} {before_score:.2f} / 5.0",
            f"수정 후(2차) | {bar(after_score):<40} {after_score:.2f} / 5.0",
            f"변화량       | {after_score - before_score:+.2f}점",
        ]
    )


def delta_str(delta: int) -> str:
    return f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "0")


def render_report(
    analysis: dict[str, Any],
    before_dist: dict[int, int],
    after_dist: dict[int, int],
    after_keywords: list[tuple[str, int]],
    citizen_rows: list[dict[str, Any]],
    reflection: dict[str, Any],
) -> str:
    policy_name = analysis["policy_name"]
    before_score = analysis["before_score"]
    after_score = analysis["after_score"]
    delta = analysis["delta_score"]
    change = analysis["opinion_change"]
    lines: list[str] = []

    # 표지
    lines += [
        "# 화순군",
        "",
        "# 2026 화순군 지역정책 가상시민 반응 시뮬레이션",
        "## 2차 결과보고서",
        "",
        "---",
        "",
    ]

    # 개요
    lines += [
        "# 개요",
        "",
        f"- **정책명** : {policy_name}",
        "- **분석단위** : 화순군 가상시민 100명",
        "- **분석내용** :",
        "    - 수정 후 만족도",
        "    - 수정 후 시민 의견",
        "    - 정책 변경 전후 비교",
        "    - 개선사항 반영 정도",
        "",
        f"- **수정 전 만족도** : {before_score:.2f} / 5.0점",
        f"- **수정 후 만족도** : {after_score:.2f} / 5.0점",
        f"- **평균 만족도 변화량** : {delta:+.2f}점",
        f"- **개선사항 반영률(CRR)** : {reflection['crr']:.1f}%",
        "",
        "---",
        "",
    ]

    # Ⅰ. 보고 목적 및 분석 개요
    lines += ["# Ⅰ. 보고 목적 및 분석 개요", "", "## 1. 보고 목적", ""]
    lines += [f"○ {line}" for line in FIXED_PURPOSE_LINES]
    lines += ["", "## 2. 분석 개요", ""]
    lines += [f"○ {line}" for line in FIXED_ANALYSIS_OVERVIEW]
    lines += ["", "---", ""]

    # Ⅱ. 수정 정책 시나리오
    lines += [
        "# Ⅱ. 수정 정책 시나리오",
        "",
        analysis["revised_policy_summary"],
        "",
        "---",
        "",
    ]

    # Ⅲ. 가상시민 재평가 기준
    lines += [
        "# Ⅲ. 가상시민 재평가 기준",
        "",
        "## 1. 재평가 대상",
        "",
        FIXED_REEVAL_TARGET,
        "",
        "## 2. 개선사항 반영 기준",
        "",
        FIXED_REFLECTION_RULE,
        "",
        "---",
        "",
    ]

    # Ⅳ. 정책 수정 후 분석
    lines += ["# Ⅳ. 정책 수정 후 분석", "", "## 1. 총평", ""]
    lines += [f"○ {line}" for line in analysis["review_lines"]]
    lines += [
        "",
        "## 2. 만족도 분포 (수정 후)",
        "",
        "```text",
        distribution_bar_chart(after_dist),
        "```",
        "",
        "## 3. 주요 키워드 (수정 후)",
        "",
    ]
    if after_keywords:
        for rank, (keyword, count) in enumerate(after_keywords, start=1):
            lines.append(f"{rank}. {keyword} ({count}명)")
    else:
        lines.append("키워드가 집계되지 않음.")
    lines += ["", "---", ""]

    # Ⅴ. 정책 변경 전후 비교
    lines += [
        "# Ⅴ. 정책 변경 전후 비교",
        "",
        "## 1. 평균 만족도 비교",
        "",
        "```text",
        mean_bar_chart(before_score, after_score),
        "```",
        "",
        f"○ **{analysis['ttest_phrase']}**",
        "",
        f"○ {analysis['ttest_detail']}",
        "",
        "## 2. 만족도 분포 비교",
        "",
        "```text",
        comparison_bar_chart(before_dist, after_dist),
        "```",
        "",
        "## 3. 의견 변화 분석",
        "",
        f"○ 만족도 상승 {change['improved_count']}명 · 하락 {change['worsened_count']}명 · "
        f"변화 없음 {change['unchanged_count']}명.",
        "",
        f"- **긍정 증가 요인** : {', '.join(change['positive_drivers']) or '뚜렷한 요인 없음'}",
        f"- **부정 감소 요인** : {', '.join(change['negative_reducers']) or '뚜렷한 요인 없음'}",
        f"- **새롭게 등장한 우려** : {', '.join(change['new_concerns']) or '없음'}",
        f"- **사라진 우려** : {', '.join(change['resolved_concerns']) or '없음'}",
        "",
        "---",
        "",
    ]

    # Ⅵ. 개선사항 반영 분석
    lines += [
        "# Ⅵ. 개선사항 반영 분석",
        "",
        "○ 1차 보고서에서 제시한 개선 필요사항이 수정 정책에 얼마나 반영되었는지 분석함.",
        "",
    ]
    if reflection["items"]:
        lines += [
            "| 개선 요구사항 | 반영 여부 | 점수 | 판단 근거 |",
            "| --- | --- | --- | --- |",
        ]
        for item in reflection["items"]:
            title = shorten(item["title"], 60).replace("|", "/")
            reason = shorten(item["reason"], 80).replace("|", "/")
            lines.append(
                f"| {title} | {VERDICT_KR[item['verdict']]} ({item['verdict']}) | "
                f"{item['score']:.1f} | {reason} |"
            )
        lines += [
            "",
            "### 종합 결과",
            "",
            f"- 전체 개선사항 : {reflection['total']}건",
            f"- 완전 반영(Resolved) : {reflection['resolved']}건",
            f"- 부분 반영(Partially Resolved) : {reflection['partially']}건",
            f"- 미반영(Unresolved) : {reflection['unresolved']}건",
            f"- 점수 총합 : {reflection['score_sum']} / {reflection['total']}",
            f"- **개선사항 반영률(CRR)** : {reflection['crr']:.1f}%",
        ]
    else:
        lines.append(
            "1차 보고서의 개선 필요사항(improvements)이 입력되지 않아 반영 분석을 생략함. "
            "`--previous-report`로 report_data.json을 지정하면 자동 분석됨."
        )
    lines += ["", "---", ""]

    # 부록 1. 동일 시민 100명 전/후 비교
    lines += [
        "# 부록 1. 동일 가상시민 전/후 비교",
        "",
        f"○ 아래 표는 1차와 동일한 가상시민 {len(citizen_rows)}명을 ID 기준으로 1:1 비교한 결과임.",
        "",
        "| ID | 수정 전 | 수정 후 | 변화량 | 키워드 | 수정 후 의견 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in citizen_rows:
        keyword_text = ", ".join(row["after_keywords"]) or "키워드 없음"
        opinion = shorten(natural_polite(row["after_opinion"]), 120).replace("|", "/")
        lines.append(
            f"| {row['id']} | {row['before_score']}점 | {row['after_score']}점 | "
            f"{delta_str(row['delta'])}점 | {keyword_text} | {opinion} |"
        )
    # 부록 2(대표 시민 토론)는 제거함 — 2차 보고서는 토론 없이 100명 만족도와
    # 유의성(t-test)만 다룬다. 토론은 1차(토론.py)의 역할이다.

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# HTML 렌더링(더블클릭으로 열람 가능한 자체 완결형 리포트)
# ---------------------------------------------------------------------------
def _esc(text: Any) -> str:
    import html

    return html.escape(str(text if text is not None else ""))


def _bar_chart_html(distribution: dict[int, int]) -> str:
    """토론.py와 동일한 만족도 막대(전체 100명 대비 %, 점수별 의미색)."""
    total = sum(distribution.values()) or 1
    rows = []
    for score in range(1, 6):
        count = distribution.get(score, 0)
        ratio = count / total * 100  # 라벨과 동일 값이 곧 막대 width
        color = _SCORE_COLORS.get(score, "#8b8d98")
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{score}점 <span>{_esc(SCORE_LABELS.get(score, ""))}</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{ratio:.1f}%;background:{color}"></div></div>'
            f'<div class="bar-value">{count}명 <span>({ratio:.1f}%)</span></div>'
            f"</div>"
        )
    return '<div class="bar-chart">' + "".join(rows) + "</div>"


def _kw_cloud_html(after_keywords: list[tuple[str, int]]) -> str:
    """토론.py와 동일한 키워드 클라우드(순위 강조는 글자 크기, opacity 하한 0.7)."""
    if not after_keywords:
        return '<p class="muted">키워드가 집계되지 않음.</p>'
    chips = []
    for rank, (keyword, count) in enumerate(after_keywords):
        size = 1.55 - rank * 0.16
        opacity = max(0.7, 1.0 - rank * 0.1)
        chips.append(
            f'<span class="kw" style="font-size:{size:.2f}rem;opacity:{opacity:.2f}">'
            f'<b>{rank + 1}</b> {_esc(keyword)} <em>{count}명</em></span>'
        )
    return '<div class="kw-cloud">' + "".join(chips) + "</div>"


def render_report_html(
    analysis: dict[str, Any],
    before_dist: dict[int, int],
    after_dist: dict[int, int],
    after_keywords: list[tuple[str, int]],
    citizen_rows: list[dict[str, Any]],
    reflection: dict[str, Any],
    ttest: dict[str, Any],
) -> str:
    policy_name = analysis["policy_name"]
    before_score = analysis["before_score"]
    after_score = analysis["after_score"]
    delta = analysis["delta_score"]
    change = analysis["opinion_change"]

    # 만족도 게이지 채움: (score-1)/4 로 1~5점을 0~100%에 매핑(토론.py와 동일 규칙).
    def gauge_pct(score: float) -> float:
        return max(0.0, min(100.0, (score - 1) / 4 * 100))

    sig_class = "sig-up" if (ttest["significant"] and ttest["direction"] == "increase") else (
        "sig-down" if (ttest["significant"] and ttest["direction"] == "decrease") else "sig-none"
    )
    delta_cls = "up" if delta > 0 else ("down" if delta < 0 else "")

    review_html = "".join(
        f'<div class="fixed">○ {_esc(line)}</div>' for line in analysis["review_lines"]
    )

    # Ⅱ 수정 정책 시나리오 — 반영된 개선사항 칩(수정된 내용 강조).
    reflected = analysis.get("reflected_improvements", [])
    reflected_html = (
        '<h3>이번 수정에 반영된 개선사항</h3><div class="kw-cloud">'
        + "".join(f'<span class="kw">{_esc(shorten(t, 40))}</span>' for t in reflected)
        + "</div>"
        if reflected
        else ""
    )

    # Ⅵ 개선사항 반영 — 토론.py의 improve-item 카드 스타일.
    if reflection["items"]:
        improvements_html = "".join(
            f'<div class="improve-item"><div class="improve-no {it["verdict"].split()[0].lower()}">'
            f'{it["score"]:.1f}</div>'
            f'<div class="improve-body"><div class="improve-title">{_esc(shorten(it["title"], 70))} '
            f'<span class="verdict-badge {it["verdict"].split()[0].lower()}">{_esc(VERDICT_KR[it["verdict"]])}</span></div>'
            f'<div class="improve-reason">{_esc(shorten(it["reason"], 110))}</div></div></div>'
            for it in reflection["items"]
        )
        crr_html = (
            f'<div class="fixed">○ 개선사항 반영률(CRR) : <b>{reflection["crr"]:.1f}%</b> '
            f'(완전 {reflection["resolved"]} · 부분 {reflection["partially"]} · '
            f'미반영 {reflection["unresolved"]} / 전체 {reflection["total"]}건)</div>'
        )
    else:
        improvements_html = '<p class="muted">1차 개선 필요사항이 입력되지 않아 반영 분석을 생략함.</p>'
        crr_html = ""

    # 부록 1 — 동일 시민 100명 전/후 비교 표(토론.py 표 스타일).
    def _kw_cell(keywords: list[str]) -> str:
        return "".join(f'<span class="kw-tag">{_esc(k)}</span>' for k in keywords) or "—"

    def _row_html(r: dict[str, Any]) -> str:
        before_c = _SCORE_COLORS.get(r["before_score"], "#8b8d98")
        after_c = _SCORE_COLORS.get(r["after_score"], "#8b8d98")
        d_cls = "up" if r["delta"] > 0 else ("down" if r["delta"] < 0 else "flat")
        return (
            f"<tr><td class='cid'>{_esc(r['id'])}</td>"
            f"<td class='score-cell'><span class='score-pill' style='background:{before_c}'>{r['before_score']}</span></td>"
            f"<td class='score-cell'><span class='score-pill' style='background:{after_c}'>{r['after_score']}</span></td>"
            f"<td class='score-cell delta-{d_cls}'>{delta_str(r['delta'])}</td>"
            f"<td class='kw-cell'>{_kw_cell(r['after_keywords'])}</td>"
            f"<td class='opinion'>{_esc(shorten(natural_polite(r['after_opinion']), 110))}</td></tr>"
        )

    citizen_rows_html = "".join(_row_html(r) for r in citizen_rows)

    analysis_items = "".join(
        f"<li>{_esc(item)}</li>"
        for item in ("수정 후 만족도", "수정 후 시민 의견", "정책 변경 전후 비교", "개선사항 반영 정도")
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(policy_name)} 가상시민 반응 시뮬레이션 2차 보고서</title>
<style>
  :root {{ --blue:#2563eb; --blue-d:#1e40af; --ink:#1a1d1f; --muted:#6b7280; --line:#e5e7eb; --bg:#f5f6f7; --card:#ffffff;
           --up:#16a34a; --down:#dc2626; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:'Pretendard','Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR',sans-serif;
    line-height:1.7; -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:0 20px 80px; }}
  .cover {{ background:linear-gradient(135deg,var(--blue),var(--blue-d)); color:#fff;
    padding:56px 40px; border-radius:0 0 24px 24px; text-align:center; box-shadow:0 8px 24px rgba(0,0,0,.12); }}
  .cover .badge {{ display:inline-block; background:rgba(255,255,255,.18); padding:6px 16px; border-radius:999px;
    font-size:.85rem; letter-spacing:.05em; margin-bottom:18px; }}
  .cover h1 {{ margin:.2em 0; font-size:1.9rem; font-weight:800; }}
  .cover .sub {{ font-size:1.05rem; opacity:.92; }}
  section {{ background:var(--card); border:1px solid var(--line); border-radius:16px;
    padding:28px 32px; margin-top:22px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
  h2 {{ font-size:1.35rem; margin:0 0 18px; padding-bottom:12px; border-bottom:2px solid var(--blue);
    color:var(--blue-d); display:flex; align-items:center; gap:10px; }}
  h3 {{ font-size:1.08rem; margin:22px 0 10px; color:#111; }}
  .fixed {{ background:#eff4ff; border-left:4px solid var(--blue); padding:12px 16px; border-radius:8px;
    margin:10px 0; color:#25366b; }}
  .fixed.sig-up {{ background:#ecfdf5; border-left-color:var(--up); color:#166534; font-weight:700; }}
  .fixed.sig-down {{ background:#fef2f2; border-left-color:var(--down); color:#991b1b; font-weight:700; }}
  .fixed.sig-none {{ background:#f1f5f9; border-left-color:#64748b; color:#334155; font-weight:700; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .meta-row {{ display:flex; gap:12px; padding:8px 0; border-bottom:1px dashed var(--line); }}
  .meta-row .k {{ min-width:110px; color:var(--muted); font-weight:600; }}
  ul {{ margin:8px 0; padding-left:20px; }}
  li {{ margin:4px 0; }}
  .muted {{ color:var(--muted); }}
  /* 종합 만족도 배너(전/후 비교) */
  .score-banner {{ display:flex; align-items:center; gap:28px; background:linear-gradient(135deg,#eff4ff,#e3ebff);
    border-radius:14px; padding:24px 28px; }}
  .score-big {{ font-size:3.4rem; font-weight:800; color:var(--blue-d); line-height:1; }}
  .score-big small {{ font-size:1.1rem; color:var(--muted); font-weight:600; }}
  .score-big .delta {{ display:block; font-size:1.1rem; margin-top:8px; }}
  .score-big .delta.up {{ color:var(--up); }} .score-big .delta.down {{ color:var(--down); }}
  .score-meter {{ flex:1; }}
  .cmp-row {{ display:flex; align-items:center; gap:12px; margin:6px 0; font-size:.9rem; }}
  .cmp-lab {{ min-width:80px; color:var(--muted); font-weight:600; }}
  .cmp-row .track {{ flex:1; height:14px; background:#d7e0f5; border-radius:999px; overflow:hidden; }}
  .cmp-row .fill {{ height:100%; background:linear-gradient(90deg,var(--blue),var(--blue-d)); border-radius:999px; }}
  .cmp-row .fill.after {{ background:linear-gradient(90deg,#22c55e,#16a34a); }}
  .cmp-val {{ min-width:44px; text-align:right; font-variant-numeric:tabular-nums; font-weight:700; }}
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
  .kw {{ background:#eef3ff; color:var(--blue-d); padding:8px 16px; border-radius:999px; font-weight:600; }}
  .kw b {{ color:var(--blue); margin-right:4px; }}
  .kw em {{ font-style:normal; color:var(--muted); font-size:.82rem; margin-left:4px; }}
  /* 의견 변화 요약 카드 */
  .stance-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
  .stance-card {{ background:#fafbfc; border:1px solid var(--line); border-radius:12px; padding:16px; text-align:center; }}
  .stance-card .n {{ font-size:1.8rem; font-weight:800; }}
  .stance-card .n.up {{ color:var(--up); }} .stance-card .n.down {{ color:var(--down); }}
  .stance-card .k {{ font-size:.9rem; color:var(--muted); margin-top:4px; }}
  /* 개선사항 반영 카드 */
  .improve-item {{ display:flex; gap:16px; padding:16px; background:#fafbfc; border:1px solid var(--line);
    border-radius:12px; margin-bottom:12px; }}
  .improve-no {{ flex:none; width:44px; height:44px; border-radius:50%; background:var(--blue); color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.95rem; }}
  .improve-no.resolved {{ background:var(--up); }} .improve-no.partially {{ background:#f59e0b; }}
  .improve-no.unresolved {{ background:#94a3b8; }}
  .improve-title {{ font-weight:700; margin-bottom:4px; }}
  .improve-reason {{ font-size:.9rem; color:var(--muted); }}
  .verdict-badge {{ font-size:.72rem; font-weight:700; color:#fff; padding:2px 9px; border-radius:999px; margin-left:6px; }}
  .verdict-badge.resolved {{ background:var(--up); }} .verdict-badge.partially {{ background:#f59e0b; }}
  .verdict-badge.unresolved {{ background:#94a3b8; }}
  /* 표 */
  .table-scroll {{ max-height:560px; overflow:auto; border:1px solid var(--line); border-radius:12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  thead th {{ position:sticky; top:0; background:var(--blue-d); color:#fff; padding:10px 12px; text-align:left; z-index:1; }}
  td {{ padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
  tbody tr:nth-child(even) {{ background:#fafbfc; }}
  .cid {{ font-weight:600; white-space:nowrap; }}
  .score-cell {{ text-align:center; }}
  .score-pill {{ display:inline-block; min-width:26px; padding:2px 8px; border-radius:999px; color:#fff; font-weight:700; }}
  .score-cell.delta-up {{ color:var(--up); font-weight:800; }}
  .score-cell.delta-down {{ color:var(--down); font-weight:800; }}
  .kw-cell {{ display:flex; flex-wrap:wrap; gap:4px; }}
  .kw-tag {{ display:inline-block; background:#eef3ff; color:var(--blue-d); padding:1px 8px; border-radius:999px;
    font-size:.75rem; font-weight:600; }}
  .opinion {{ color:#374151; }}
  footer {{ text-align:center; color:var(--muted); font-size:.85rem; margin-top:32px; }}
  @media (max-width:680px) {{ .grid2,.stance-grid {{ grid-template-columns:1fr; }} .score-banner {{ flex-direction:column; text-align:center; }} }}
</style>
</head>
<body>
<div class="cover">
  <div class="badge">화순군 · 가상시민 반응 시뮬레이션</div>
  <h1>{_esc(policy_name)} 정책 수정 전·후 비교</h1>
  <div class="sub">2차 결과보고서</div>
</div>
<div class="wrap">

  <section>
    <h2>개요</h2>
    <div class="meta-row"><span class="k">정책명</span><span>{_esc(policy_name)}</span></div>
    <div class="meta-row"><span class="k">분석단위</span><span>화순군 가상시민 100명 (1차와 동일)</span></div>
    <div class="meta-row"><span class="k">분석내용</span><span>수정 후 만족도 · 수정 후 시민 의견 · 정책 변경 전후 비교 · 개선사항 반영 정도</span></div>
    <h3>만족도 변화</h3>
    <div class="score-banner">
      <div class="score-big">{after_score:.2f}<small> / 5.0점</small>
        <span class="delta {delta_cls}">수정 전 대비 {delta:+.2f}점</span></div>
      <div class="score-meter">
        <div class="cmp-row"><span class="cmp-lab">수정 전(1차)</span>
          <div class="track"><div class="fill" style="width:{gauge_pct(before_score):.1f}%"></div></div>
          <span class="cmp-val">{before_score:.2f}</span></div>
        <div class="cmp-row"><span class="cmp-lab">수정 후(2차)</span>
          <div class="track"><div class="fill after" style="width:{gauge_pct(after_score):.1f}%"></div></div>
          <span class="cmp-val">{after_score:.2f}</span></div>
        <p class="muted" style="margin:10px 0 0">개선사항 반영률(CRR) {reflection['crr']:.1f}%</p>
      </div>
    </div>
  </section>

  <section>
    <h2>Ⅰ. 보고 목적 및 분석 개요</h2>
    <h3>1. 보고 목적</h3>
    <div class="fixed">○ {_esc(FIXED_PURPOSE_LINES[0])}</div>
    <div class="fixed">○ {_esc(FIXED_PURPOSE_LINES[1])}</div>
    <h3>2. 분석 개요</h3>
    <div class="grid2">
      <div><div class="meta-row"><span class="k">분석 대상</span><span>{_esc(policy_name)}</span></div>
        <div class="meta-row"><span class="k">가상 시민</span><span>1차와 동일한 가상시민 100명</span></div></div>
      <div><b>출력 항목</b><ul>{analysis_items}</ul></div>
    </div>
  </section>

  <section>
    <h2>Ⅱ. 수정 정책 시나리오</h2>
    <p>{_esc(analysis['revised_policy_summary'])}</p>
    {reflected_html}
  </section>

  <section>
    <h2>Ⅲ. 가상시민 재평가 기준</h2>
    <h3>1. 재평가 대상</h3><p>{_esc(FIXED_REEVAL_TARGET.lstrip('○ '))}</p>
    <h3>2. 개선사항 반영 기준</h3><p>{_esc(FIXED_REFLECTION_RULE.lstrip('○ '))}</p>
  </section>

  <section>
    <h2>Ⅳ. 정책 수정 후 분석</h2>
    <h3>1. 총평</h3>
    {review_html}
    <h3>2. 만족도 분포 (수정 후)</h3>
    {_bar_chart_html(after_dist)}
    <h3>3. 주요 키워드 (수정 후)</h3>
    {_kw_cloud_html(after_keywords)}
  </section>

  <section>
    <h2>Ⅴ. 정책 변경 전후 비교</h2>
    <h3>1. 만족도 통계 검정 (t-test)</h3>
    <div class="fixed {sig_class}">○ {_esc(analysis['ttest_phrase'])}</div>
    <p class="muted" style="margin:6px 0 0">{_esc(analysis['ttest_detail'])}</p>
    <h3>2. 만족도 분포 비교</h3>
    <p class="muted" style="margin:0">수정 전(1차)</p>
    {_bar_chart_html(before_dist)}
    <p class="muted" style="margin:16px 0 0">수정 후(2차)</p>
    {_bar_chart_html(after_dist)}
    <h3>3. 의견 변화 분석</h3>
    <div class="stance-grid">
      <div class="stance-card"><div class="n up">{change['improved_count']}</div><div class="k">만족도 상승(명)</div></div>
      <div class="stance-card"><div class="n down">{change['worsened_count']}</div><div class="k">하락(명)</div></div>
      <div class="stance-card"><div class="n">{change['unchanged_count']}</div><div class="k">변화 없음(명)</div></div>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div><h3 style="margin-top:0">긍정 증가 요인</h3><p>{_esc(', '.join(change['positive_drivers']) or '뚜렷한 요인 없음')}</p>
        <h3>사라진 우려</h3><p>{_esc(', '.join(change['resolved_concerns']) or '없음')}</p></div>
      <div><h3 style="margin-top:0">부정 감소 요인</h3><p>{_esc(', '.join(change['negative_reducers']) or '뚜렷한 요인 없음')}</p>
        <h3>새롭게 등장한 우려</h3><p>{_esc(', '.join(change['new_concerns']) or '없음')}</p></div>
    </div>
  </section>

  <section>
    <h2>Ⅵ. 개선사항 반영 분석</h2>
    <div class="fixed">○ 1차 보고서에서 제시한 개선 필요사항이 수정 정책에 얼마나 반영되었는지 분석함.</div>
    {improvements_html}
    {crr_html}
  </section>

  <section>
    <h2>부록 1. 동일 가상시민 전/후 비교</h2>
    <p class="muted">1차와 동일한 가상시민 {len(citizen_rows)}명을 ID 기준으로 1:1 비교한 결과입니다. (표 안에서 스크롤됩니다)</p>
    <div class="table-scroll">
      <table><thead><tr><th>ID</th><th>수정 전</th><th>수정 후</th><th>변화량</th><th>키워드</th><th>수정 후 의견</th></tr></thead>
      <tbody>{citizen_rows_html}</tbody></table>
    </div>
  </section>

  <footer>화순군 가상시민 반응 시뮬레이션 · 2차 결과보고서 · 자동 생성 (본 결과는 시뮬레이션이며 실제 설문이 아님)</footer>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run_after_simulation(args: argparse.Namespace, revised_policy_text: str) -> dict[str, Any]:
    simulator = HwasunPersonaPolicySimulator(
        persona_path=args.persona_path,
        db_path=args.db_path,
        output_path=args.after_output,
        model=args.model,
        use_api=not args.no_api,
        api_key_env=args.api_key_env,
    )
    return simulator.run(
        policy_text=revised_policy_text,
        persona_limit=args.persona_limit,
        rag_limit=args.rag_limit,
        print_rows=not args.quiet,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="정책 수정 후(2차) 재시뮬레이션 및 1차 대비 전/후 비교 보고서 생성"
    )
    # 수정 정책 입력(재시뮬레이션용)
    parser.add_argument("--policy-text", help="수정된 정책 텍스트 직접 입력")
    parser.add_argument(
        "--policy-file",
        default=str(DEFAULT_REVISED_POLICY_PATH),
        help="수정된 정책 텍스트 파일 경로",
    )
    # 1차 결과 / 1차 보고서
    parser.add_argument("--before", default=str(DEFAULT_BEFORE_PATH), help="1차(general_100) 결과 JSON 경로")
    parser.add_argument(
        "--previous-report",
        default=str(DEFAULT_PREVIOUS_REPORT),
        help="1차 보고서 report_data.json 경로(개선사항·대표 시민)",
    )
    # 2차 결과: 새로 돌리거나(기본) 기존 파일 사용
    parser.add_argument("--after-file", default="", help="이미 돌려둔 2차 결과 JSON(지정 시 재시뮬레이션 생략)")
    parser.add_argument("--after-output", default=str(DEFAULT_AFTER_OUTPUT), help="2차 시뮬레이션 결과 저장 경로")
    # 보고서 출력
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT), help="2차 비교 보고서 마크다운 경로")
    parser.add_argument("--html-output", default=str(DEFAULT_HTML_OUTPUT), help="2차 비교 보고서 HTML 경로(더블클릭 열람용)")
    parser.add_argument("--data-output", default=str(DEFAULT_DATA_OUTPUT), help="비교 데이터 묶음 JSON 경로")
    # general_100 시뮬레이션 파라미터
    parser.add_argument("--persona-path", default=str(DEFAULT_PERSONA_PATH))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--persona-limit", type=int, default=None, help="테스트용 페르소나 수 제한")
    parser.add_argument("--rag-limit", type=int, default=3)
    parser.add_argument("--no-api", action="store_true", help="OpenAI API 없이 로컬 휴리스틱으로 2차 결과 생성")
    parser.add_argument("--quiet", action="store_true", help="100명 개별 출력 없이 파일만 저장")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_arg_parser().parse_args(argv)

    before_path = Path(args.before).expanduser().resolve()
    if not before_path.exists():
        print(f"실행 중단: 1차 결과 파일을 찾을 수 없습니다: {before_path}", file=sys.stderr)
        return 1
    before_payload = load_simulation_payload(before_path)
    before_results = before_payload["results"]

    previous_report_path = (
        Path(args.previous_report).expanduser().resolve() if args.previous_report else None
    )

    # 2차 결과 확보: 파일 재사용 또는 재시뮬레이션.
    if args.after_file:
        after_path = Path(args.after_file).expanduser().resolve()
        if not after_path.exists():
            print(f"실행 중단: 2차 결과 파일을 찾을 수 없습니다: {after_path}", file=sys.stderr)
            return 1
        after_payload = load_simulation_payload(after_path)
        revised_policy_text = args.policy_text or (
            read_text_auto(Path(args.policy_file)) if args.policy_file else ""
        )
    else:
        try:
            revised_policy_text = read_revised_policy_text(args)
        except ValueError as exc:
            print(f"실행 중단: {exc}", file=sys.stderr)
            return 1
        try:
            after_payload = run_after_simulation(args, revised_policy_text)
        except RuntimeError as exc:
            print(f"실행 중단: {exc}", file=sys.stderr)
            print("API 없이 구조만 테스트하려면 --no-api를, 재시뮬레이션을 건너뛰려면 --after-file을 사용하세요.", file=sys.stderr)
            return 1

    after_results = after_payload["results"]
    revised_policy = after_payload.get("policy") or {}

    # 수정 정책 텍스트 확보(개선사항 반영 판정용). 없으면 정책 dict에서 재구성.
    if not revised_policy_text:
        parts = [
            clean_space(revised_policy.get("policy_name")),
            clean_space(revised_policy.get("main_details")),
            clean_space(revised_policy.get("target_audience")),
            " ".join(str(k) for k in (revised_policy.get("search_keywords") or [])),
        ]
        revised_policy_text = " ".join(part for part in parts if part)

    before_idx = result_index(before_results)
    after_idx = result_index(after_results)
    citizen_rows = build_citizen_comparison(before_idx, after_idx)
    if not citizen_rows:
        print("실행 중단: 1차·2차 결과에서 공통 시민 ID를 찾지 못했습니다(동일 페르소나 여부 확인).", file=sys.stderr)
        return 1

    improvements = load_improvements(previous_report_path, before_payload)
    reflection = build_reflection_analysis(improvements, revised_policy_text)

    before_dist = score_distribution(before_results)
    after_dist = score_distribution(after_results)
    after_keywords = keyword_counter(after_results).most_common(5)

    # 동일 시민 100명 대응표본 t-검정(수정 전/후 만족도 유의차 검정).
    ttest = paired_ttest(citizen_rows)

    analysis = build_analysis(
        revised_policy=revised_policy,
        revised_policy_text=revised_policy_text,
        before_results=before_results,
        after_results=after_results,
        citizen_rows=citizen_rows,
        reflection=reflection,
        ttest=ttest,
    )

    report_md = render_report(
        analysis=analysis,
        before_dist=before_dist,
        after_dist=after_dist,
        after_keywords=after_keywords,
        citizen_rows=citizen_rows,
        reflection=reflection,
    )

    report_html = render_report_html(
        analysis=analysis,
        before_dist=before_dist,
        after_dist=after_dist,
        after_keywords=after_keywords,
        citizen_rows=citizen_rows,
        reflection=reflection,
        ttest=ttest,
    )

    report_path = Path(args.report_output).expanduser().resolve()
    html_path = Path(args.html_output).expanduser().resolve()
    data_path = Path(args.data_output).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding=UTF8_SIG)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    # HTML은 브라우저 더블클릭 호환을 위해 BOM 없는 UTF-8로 저장.
    html_path.write_text(report_html, encoding="utf-8")
    write_json(
        data_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "policy_name": analysis["policy_name"],
            "before_score": analysis["before_score"],
            "after_score": analysis["after_score"],
            "delta_score": analysis["delta_score"],
            "reflection_rate": reflection["crr"],
            "ttest": ttest,
            "before_distribution": {str(k): v for k, v in before_dist.items()},
            "after_distribution": {str(k): v for k, v in after_dist.items()},
            "after_keywords": [{"keyword": k, "count": c} for k, c in after_keywords],
            "opinion_change": analysis["opinion_change"],
            "reflection": reflection,
            "citizen_comparison": citizen_rows,
        },
    )

    print("\n" + "=" * 52)
    print(" 정책 수정 전/후 비교(2차 보고서) 생성 완료")
    print("=" * 52)
    print(f" 정책명            : {analysis['policy_name']}")
    print(f" 수정 전 만족도    : {analysis['before_score']:.2f} / 5.0")
    print(f" 수정 후 만족도    : {analysis['after_score']:.2f} / 5.0")
    print(f" 평균 변화량       : {analysis['delta_score']:+.2f}점")
    print(f" t-검정            : t({ttest['df']})={ttest['t_stat']:.3f}, p={ttest['p_value']:.4f} "
          f"→ {'유의미' if ttest['significant'] else '비유의미'}")
    print(f" 판정 문구         : {analysis['ttest_phrase']}")
    print(f" 개선사항 반영률   : {reflection['crr']:.1f}%  "
          f"(완전 {reflection['resolved']}/부분 {reflection['partially']}/미반영 {reflection['unresolved']})")
    print(f" 비교 대상 시민 수 : {len(citizen_rows)}명")
    print("-" * 52)
    print(f" - 2차 보고서 : {report_path}")
    print(f" - HTML 보고서: {html_path}")
    print(f" - 비교 데이터: {data_path}")
    if not args.after_file:
        print(f" - 2차 원본   : {Path(args.after_output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
