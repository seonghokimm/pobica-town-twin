# -*- coding: utf-8 -*-
"""
evidence_pack_builder.py
Town-Twin RAG(sqlite3)에서 대표 에이전트별 '증거팩'을 조립하는 프로토타입.

사용: python evidence_pack_builder.py --db town_twin_full_v3_policy_stage.sqlite3 \
        --emd 백아면 --issue 접근성 --persona-tags 고령,이동제약
출력: evidence_pack_<읍면>_<쟁점>.json
"""
import sqlite3, json, argparse, re

# ── 읍면 → bus_routes.path에서 찾을 정류장/옛지명 키워드 ──
# path 컬럼이 옛 지명·마을 정류장명으로 적혀 있어 읍면명 직접 매칭 불가.
# 2026-07-09 DB 전수조사(bus_routes 45행 + minwon + persona_seed 교차확인)로 검증 완료.
EMD_STOP_KEYWORDS = {
    "백아면": ["북면", "송단", "유천리", "노치", "하다", "가수3"],
    "화순읍": ["화순", "삼천리"],           # 삼천리=화순읍 소속, minwon으로 확인
    "동복면": ["동복"],
    "사평면": ["사평", "유마사", "절산"],
    "이서면": ["이서", "야사"],
    "능주면": ["능주"],
    "도곡면": ["도곡"],
    "이양면": ["이양"],
    "동면": ["서성"],                        # 동면 서성리 민원과 215-1 노선 교차 확인
    "도암면": ["등광"],                      # 등광리=도암면 소재지 (218번), 웹검색으로 확인
    "청풍면": ["곰치"],                      # 곰치휴게소=청풍면 이만리 (218-1번 종점), 웹검색 확인
    "춘양면": ["우봉"],                      # 우봉리='춘양 계열' 정류장 (218번), 나무위키 확인
    "한천면": ["오음"],                      # 오음2·3리(한천자연휴양림)='한천 계열' (218번), 나무위키 확인
}

# 도암면: bus_routes(등광 키워드)로 노선 근거 확보 + 민원에서 마을 단위 구체 수치도 확보
# (둘 다 유효한 근거라 병행 사용. 노선 전체 빈도와 옥동마을 개별 체감은 다를 수 있어 구분해서 카드화)
SUPPLEMENTARY_MINWON_NOTES = {
    "도암면": "화순군 도암면 원천리 옥동마을 군내버스 관련 민원(rag_documents minwon)에 "
              "'하루 현재 3회 운행'이라는 마을 단위 체감 수치가 있음 — "
              "218번 노선 전체 운행횟수와는 별개로, 이 마을 개별 체감으로 인용할 것.",
}

# 웹검색·나무위키 교차 확인으로도 매칭 정류장을 못 찾은 읍면이 남으면 여기에 등록
NO_TRANSIT_DATA_EMD: list[str] = []

# ── [수작업 2] 쟁점(군집명) → 민원 issue_domain / 조례·지표 키워드 ──
ISSUE_MAP = {
    "접근성": {
        "minwon_domains": ["교통/도로/주차"],
        "keywords": ["버스", "교통", "노선", "이동", "정류장", "멀어", "불편"],
        "indicator_kw": ["총인구", "고령", "65세"],
    },
    "취업지원": {
        "minwon_domains": ["지역경제/일자리"],
        "keywords": ["일자리", "취업", "구직", "채용", "훈련"],
        "indicator_kw": ["취업", "사업체", "종사자"],
    },
    "노인복지": {
        "minwon_domains": ["복지/보건/돌봄"],  # 실제 minwon issue_domain 값과 일치시킴(기존 "복지/취약계층"은 DB에 없는 값이었음)
        "keywords": ["노인", "어르신", "경로", "복지", "건강"],
        "indicator_kw": ["고령", "65세", "노인"],
    },
    "교육": {
        "minwon_domains": ["교육/청소년"],
        "keywords": ["교육", "학교", "학생", "장학", "입학", "청소년"],
        "indicator_kw": ["학교", "초등학교", "학생"],
    },
}


import re

# village_knowledge 12개 문서 중 실제 격차 서술이 담긴 10개 (0001=서문, 0012=주의사항은 별도 처리)
VILLAGE_KNOWLEDGE_TOPIC_IDS = [
    "village_knowledge:VK-0002", "village_knowledge:VK-0003", "village_knowledge:VK-0004",
    "village_knowledge:VK-0005", "village_knowledge:VK-0006", "village_knowledge:VK-0007",
    "village_knowledge:VK-0008", "village_knowledge:VK-0009", "village_knowledge:VK-0010",
    "village_knowledge:VK-0011",
]
VILLAGE_KNOWLEDGE_CAUTION_ID = "village_knowledge:VK-0012"


def village_gap_cards(cur, emd: str, max_cards: int = 4) -> list[dict]:
    """village_knowledge 문서를 문장 단위로 쪼개 emd가 언급된 문장만 그대로 인용.
    계산·요약 없이 '이미 만들어진 격차 문장'을 재사용하는 것이 핵심."""
    cards = []
    placeholders = ",".join("?" for _ in VILLAGE_KNOWLEDGE_TOPIC_IDS)
    rows = cur.execute(
        f"SELECT rag_id, title, content FROM rag_documents WHERE rag_id IN ({placeholders})",
        VILLAGE_KNOWLEDGE_TOPIC_IDS).fetchall()
    hits = []
    for r in rows:
        # " - `[...]` " 마커 앞에서 문장을 쪼갬
        bullets = re.split(r"(?=- `\[)", r["content"])
        for b in bullets:
            b = b.strip(" -")
            if emd in b:
                hits.append((r["title"], b))
    # emd가 직접 언급된 문장을 우선, 그 다음은 스킵 (화순군 전체 baseline은 카드 폭발 방지 위해 제외)
    for title, sentence in hits[:max_cards]:
        cards.append({
            "종류": "생활격차지표",
            "내용": f"[{title}] {sentence}",
            "출처": "rag_documents village_knowledge (이미 계산된 격차 서술, 재계산 없이 인용)",
            "함의": "타 읍면 대비 인프라 격차 — 정책의 공간적 형평성 반박/보완 근거로 사용",
        })
    # 데이터 정합성 주의사항도 emd 관련이면 별도로 첨부 (창작 방지용 경고 카드)
    caution = cur.execute(
        "SELECT content FROM rag_documents WHERE rag_id=?", (VILLAGE_KNOWLEDGE_CAUTION_ID,)).fetchone()
    if caution and emd in caution["content"]:
        for b in re.split(r"(?=- )", caution["content"]):
            if emd in b:
                cards.append({
                    "종류": "데이터주의사항",
                    "내용": b.strip(" -"),
                    "출처": "rag_documents village_knowledge (정합성 주의)",
                    "함의": "이 내용은 단정하지 말고 검증 필요 문구를 유지한 채 인용할 것",
                })
    return cards


def policy_plan_cards(cur, issue_cfg: dict, max_cards: int = 2) -> list[dict]:
    """정책계획(policy_plan) 카드. Before 단계 원칙 준수:
    metadata_json.rag_stage=='simulation_context' AND leakage_risk=='low' 인 것만 사용.
    (evaluation_reference=과거실적/문제점 등은 Before 페르소나에게 누수되면 안 됨 — 설계도 8장 원칙)"""
    keywords = issue_cfg.get("keywords", [])
    if not keywords:
        return []
    kw_cond = " OR ".join(["search_text LIKE ?"] * len(keywords))
    rows = cur.execute(
        f"SELECT rag_id, title, content, metadata_json FROM rag_documents "
        f"WHERE source_type='policy_plan' AND ({kw_cond}) LIMIT 20",
        [f"%{k}%" for k in keywords]).fetchall()
    cards = []
    for r in rows:
        meta = json.loads(r["metadata_json"] or "{}")
        if meta.get("rag_stage") != "simulation_context" or meta.get("leakage_risk") != "low":
            continue  # Before 단계 누수 방지 — evaluation_reference/high는 절대 포함하지 않음
        cards.append({
            "종류": "정책계획(Before)",
            "내용": f"[{r['title']}] " + re.sub(r"\s+", " ", r["content"])[:200] + "…",
            "출처": f"rag_documents {r['rag_id']} (rag_stage=simulation_context, leakage_risk=low)",
            "함의": "행정이 이미 계획한 사업 — '기존 계획과 겹친다/이미 하고 있었다' 류의 "
                    "반박 근거로 사용 가능. 이 사업의 실제 효과·성과는 단정하지 말 것 "
                    "(성과 정보는 evaluation_reference로 별도 분리되어 있어 여기 없음)",
        })
        if len(cards) >= max_cards:
            break
    return cards


def policy_news_cards(cur, issue_cfg: dict, max_cards: int = 1) -> list[dict]:
    """정책 관련 뉴스(13건, 소량). 이미 보도된 반응이라 '언론에도 나온 얘기'로 생생함 보강용."""
    keywords = issue_cfg.get("keywords", [])
    if not keywords:
        return []
    kw_cond = " OR ".join(["search_text LIKE ?"] * len(keywords))
    rows = cur.execute(
        f"SELECT rag_id, title, content FROM rag_documents "
        f"WHERE source_type='policy_news' AND ({kw_cond}) LIMIT {max_cards}",
        [f"%{k}%" for k in keywords]).fetchall()
    return [{
        "종류": "관련뉴스",
        "내용": f"[{r['title']}] " + re.sub(r"\s+", " ", r["content"])[:180] + "…",
        "출처": f"rag_documents {r['rag_id']}",
        "함의": "이미 언론 보도된 내용 — 여론 규모를 단정하지 말고 '보도된 사실' 수준으로만 인용",
    } for r in rows]



def _semantic_search_or_none(db_path: str, query: str, source_types: list[str],
                              top_k: int, rag_stage: str | None = None,
                              leakage_risk: str | None = None):
    """rag_sqlite_search.py의 하이브리드(임베딩+키워드) 검색 시도.
    모델 다운로드가 막혀 있거나(네트워크 제약) sentence-transformers 미설치면 None 반환
    → 호출부가 기존 LIKE 검색으로 폴백하도록 신호를 준다."""
    try:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rag_sqlite_search import search_db  # 같은 폴더에 있어야 함
        from pathlib import Path
        info, results = search_db(
            db_path=Path(db_path), query=query, top_k=top_k,
            source_types=source_types, rag_stage=rag_stage, leakage_risk=leakage_risk,
        )
        return results
    except Exception as exc:  # noqa: BLE001 - 모델 다운로드 실패 등 다양한 원인 통합 처리
        print(f"[증거팩] 하이브리드 검색 실패({type(exc).__name__}: {exc}) "
              f"→ 키워드(LIKE) 검색으로 폴백", file=__import__("sys").stderr)
        return None



def fact_card(fid, kind, content, source, implication=""):
    return {"fact_id": fid, "종류": kind, "내용": content,
            "출처": source, "함의": implication}


def build_pack(db_path, emd, issue, persona_tags):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    issue_cfg = ISSUE_MAP.get(issue, {"minwon_domains": [], "keywords": [issue], "indicator_kw": []})
    cards, cid = [], 0

    def nid(prefix):
        nonlocal cid
        cid += 1
        return f"{prefix}-{emd}-{cid:03d}"

    # ── 1) 버스 팩트카드 (SQL 정확 조회 + 코드가 요약문 생성) ──
    if emd in NO_TRANSIT_DATA_EMD:
        cards.append(fact_card(nid("BUS"), "버스노선",
                               f"{emd}은(는) bus_routes·persona_seed·village_knowledge "
                               f"전체와 웹 교차검증에도 매칭되는 정류장·노선 정보가 없음 (데이터셋 공백).",
                               "evidence_pack_builder 전수조사",
                               "대중교통 근거로 특정 수치를 발언에 넣지 말고 "
                               "'대중교통 관련 자료가 확인되지 않음'으로만 표현할 것"))
        stops = []
    else:
        stops = EMD_STOP_KEYWORDS.get(emd, [])

    seen = set()
    for kw in stops:
        # branch(세부 경로)를 우선 조회 — 정류장별 실제 첫/막차가 summary와 다를 수 있음
        branch_rows = cur.execute(
            "SELECT route_number, first_bus, last_bus, path FROM bus_routes "
            "WHERE row_type='branch' AND path LIKE ?", (f"%{kw}%",)).fetchall()
        rows = branch_rows or cur.execute(
            "SELECT route_number, first_bus, last_bus, path FROM bus_routes "
            "WHERE row_type='summary' AND path LIKE ?", (f"%{kw}%",)).fetchall()
        row_kind = "해당 정류장 경유 시간대(branch)" if branch_rows else "노선 전체 요약(summary, 다른 경유지 포함)"
        for r in rows:
            key = (r["route_number"], r["first_bus"], r["last_bus"])
            if key in seen:
                continue
            seen.add(key)
            content = (f"{r['route_number']}번 노선 — {row_kind}: {r['path']}. "
                       f"첫차 {r['first_bus']}, 막차 {r['last_bus']}. "
                       f"(개별 출발 시각표는 데이터에 없음 — 발언에서 특정 시각 창작 금지)")
            impl = ""
            try:
                span_h = (int(r['last_bus'][:2]) - int(r['first_bus'][:2]))
                if span_h < 14:
                    impl = "운행 시간대가 좁아 이른 시간을 놓치면 그날 이용이 어려움"
            except (ValueError, TypeError):
                pass
            cards.append(fact_card(nid("BUS"), "버스노선", content,
                                   f"bus_routes.route_number={r['route_number']} ({row_kind})", impl))
    if not seen and emd not in NO_TRANSIT_DATA_EMD:
        cards.append(fact_card(nid("BUS"), "버스노선",
                               f"{emd} 경유로 확인되는 노선이 bus_routes에서 검색되지 않음",
                               "bus_routes", "대중교통 근거는 '확인되지 않음'으로 발언해야 함"))

    # 노선 전체 수치 외에, 마을 단위 체감 수치가 별도로 확인된 읍면은 보충 카드 추가
    if emd in SUPPLEMENTARY_MINWON_NOTES:
        cards.append(fact_card(nid("BUS"), "버스노선(마을단위 보충)",
                               SUPPLEMENTARY_MINWON_NOTES[emd],
                               "rag_documents minwon"))

    # ── 2) 지역 지표 팩트카드 (indicator_values, region_kr 정확 조회) ──
    for ikw in issue_cfg["indicator_kw"] or ["총인구"]:
        for r in cur.execute(
            "SELECT indicator_kr, value, unit, base_period FROM indicator_values "
            "WHERE region_kr=? AND indicator_kr LIKE ? LIMIT 3", (emd, f"%{ikw}%")):
            unit = (r["unit"] or "").replace("unit: ", "")
            cards.append(fact_card(nid("IND"), "지역지표",
                                   f"{emd} {r['indicator_kr']}: {r['value']} {unit} (기준 {r['base_period']})",
                                   f"indicator_values.region_kr={emd}"))

    # ── 3) 민원 목소리 카드 (issue_domain 태그 필터 + 읍면/키워드 가점) ──
    #    region 컬럼은 96%가 '화순군'이라 못 씀 → metadata_json.issue_domain 사용
    dom_cond = " OR ".join(["json_extract(metadata_json,'$.issue_domain')=?"] * len(issue_cfg["minwon_domains"])) or "1=0"
    rows = cur.execute(
        f"SELECT rag_id, title, content FROM rag_documents "
        f"WHERE source_type='minwon' AND ({dom_cond})",
        issue_cfg["minwon_domains"]).fetchall()

    def score(row):
        s = 0
        if emd in row["content"]:
            s += 10                       # 해당 읍면 직접 언급이 최우선
        s += sum(1 for k in issue_cfg["keywords"] if k in row["content"])
        return s

    for r in sorted(rows, key=score, reverse=True)[:3]:
        cards.append(fact_card(nid("MIN"), "실제민원",
                               f"[{r['title']}] " + re.sub(r"\s+", " ", r["content"])[:200] + "…",
                               f"rag_documents.{r['rag_id']}",
                               "실제 주민이 제기한 불만 — 말투·상황 참고용(숫자 인용 금지)"))

    # ── 4) 읍면 공약 카드 (town_promise는 region이 살아있음) ──
    for r in cur.execute(
        "SELECT rag_id, content FROM rag_documents "
        "WHERE source_type='town_promise' AND region=? LIMIT 2", (emd,)):
        cards.append(fact_card(nid("PRM"), "읍면공약", r["content"],
                               f"rag_documents.{r['rag_id']}",
                               "행정이 이미 약속한 사업 — '약속과 현실의 차이'를 지적하는 반박 재료"))

    # ── 4-2) 생활격차지표 (village_knowledge, 계산 없이 이미 만들어진 문장 그대로 인용) ──
    for c in village_gap_cards(cur, emd):
        c["fact_id"] = nid("VK")
        cards.append(c)

    # ── 4-3) 정책계획 카드 (Before 단계 원칙: simulation_context/low만) ──
    # 하이브리드(임베딩+키워드) 검색 우선 시도 → 실패 시 기존 LIKE 검색 폴백
    query_text = " ".join(issue_cfg.get("keywords", []))
    semantic_plan_results = _semantic_search_or_none(
        db_path, query_text, source_types=["policy_plan"], top_k=2,
        rag_stage="simulation_context", leakage_risk="low") if query_text else None

    if semantic_plan_results is not None:
        for r in semantic_plan_results:
            cards.append(fact_card(
                nid("PLAN"), "정책계획(Before)",
                f"[{r['title']}] " + re.sub(r"\s+", " ", r["content"])[:200] + "…",
                f"rag_documents {r['rag_id']} (하이브리드 검색 score={r['score']:.3f}, "
                f"rag_stage=simulation_context, leakage_risk=low)",
                "행정이 이미 계획한 사업 — '기존 계획과 겹친다/이미 하고 있었다' 류의 "
                "반박 근거로 사용 가능. 이 사업의 실제 효과·성과는 단정하지 말 것"))
    else:
        for c in policy_plan_cards(cur, issue_cfg):  # LIKE 기반 폴백 (기존 함수 재사용)
            c["fact_id"] = nid("PLAN")
            cards.append(c)

    # ── 4-4) 관련 뉴스 카드 (13건, 소량 보강) ──
    for c in policy_news_cards(cur, issue_cfg):
        c["fact_id"] = nid("NEWS")
        cards.append(c)

    # ── 5) 조례 카드 — 하이브리드 검색 우선, 실패 시 LIKE 폴백 ──
    semantic_ord_results = _semantic_search_or_none(
        db_path, query_text, source_types=["ordinance"], top_k=2) if query_text else None

    if semantic_ord_results is not None:
        for r in semantic_ord_results:
            cards.append(fact_card(
                nid("ORD"), "조례",
                f"[{r['title']}] " + re.sub(r"\s+", " ", r["content"])[:180] + "…",
                f"rag_documents {r['rag_id']} (하이브리드 검색 score={r['score']:.3f})",
                "제도적 근거 — '조례상 근거가 있는데 왜 안 되나' 류 논증에 사용"))
    else:
        kw_cond = " OR ".join(["search_text LIKE ?"] * len(issue_cfg["keywords"]))
        for r in cur.execute(
            f"SELECT rag_id, title, content FROM rag_documents "
            f"WHERE source_type='ordinance' AND ({kw_cond}) LIMIT 2",
            [f"%{k}%" for k in issue_cfg["keywords"]]):
            cards.append(fact_card(nid("ORD"), "조례",
                                   f"[{r['title']}] " + r["content"][:180] + "…",
                                   f"rag_documents.{r['rag_id']} (폴백: 키워드 LIKE 검색)",
                                   "제도적 근거 — '조례상 근거가 있는데 왜 안 되나' 류 논증에 사용"))

    pack = {
        "대상_읍면": emd, "쟁점": issue, "페르소나_태그": persona_tags,
        "사용_규칙": [
            "발언의 모든 수치·노선·사실은 아래 카드의 fact_id를 인용해야 함",
            "카드에 없는 숫자(개별 버스 시각 등)는 창작 금지, '확인되지 않음'으로 표현",
            "실제민원 카드는 상황·말투 참고용이며 수치 인용 출처로 쓸 수 없음",
        ],
        "cards": cards,
    }
    return pack


# ── 100명 시뮬레이션 단계용 ──
# cluster_name이 아직 없는 시점(의견 생성 "전")이므로, 이미 페르소나 시드에 존재하는
# policy_relevant_traits로 이슈를 근사 추정한다. 코드(화순_0706_황*은_job_72.py)의
# SUPPORTIVE_TERMS 매핑과 동일한 방향으로 맞춤 — 완벽한 일치는 아니고 근사치임에 유의.
TRAIT_ISSUE_PRIORITY = [
    ("mobility_constraint", {"중간", "높음"}, "접근성"),
    ("income_insecurity", {"높음"}, "취업지원"),
    ("digital_accessibility", {"낮음"}, "노인복지"),  # 근사치: 정확히는 '디지털취약/복지일반'에 가까움
]


def trait_level(value) -> str:
    """policy_relevant_traits가 문자열 또는 {level, basis, source} 구조여도 level만 추출."""
    if isinstance(value, dict):
        return str(value.get("level", ""))
    return str(value)


def resolve_issue_for_100(persona: dict, policy_keywords: list[str] | None = None) -> str | None:
    """의견 생성 전 시점에 쓸 수 있는 이슈 추정. key_keywords(사후 생성)는 절대 쓰지 않음.
    우선순위: ① 정책 자체 주제 → ② 페르소나 개인 trait.
    정책이 공교육/일자리처럼 명확하면 그 맥락의 근거가 먼저 필요하고,
    정책 주제가 불명확할 때만 개인 취약성으로 보완한다."""
    traits = persona.get("policy_relevant_traits", {}) or {}
    if policy_keywords:
        joined = " ".join(policy_keywords)
        issue_hits = []
        for issue, cfg in ISSUE_MAP.items():
            hits = sum(1 for kw in cfg["keywords"] if kw in joined)
            if hits:
                issue_hits.append((hits, issue))
        if issue_hits:
            issue_hits.sort(reverse=True)
            return issue_hits[0][1]
    for trait_name, trigger_levels, issue in TRAIT_ISSUE_PRIORITY:
        if trait_level(traits.get(trait_name, "")) in trigger_levels:
            return issue
    return None  # trait도 정책 키워드도 안 맞으면 지역 기반 카드만 사용


def build_pack_for_persona(db_path: str, persona: dict,
                            policy_keywords: list[str] | None = None) -> dict:
    """100명 시뮬레이션 페르소나 1명에 대한 증거팩. build_pack()의 얇은 래퍼.
    emd/issue/tags를 persona dict에서 직접 뽑아 build_pack 인터페이스에 맞춘다.
    policy_keywords: policy.search_keywords 등 정책 자체의 키워드 (trait로 이슈를
    못 잡을 때 fallback으로 사용, 페르소나 사후생성 key_keywords와는 다름)."""
    emd = persona.get("region_name_kr") or persona.get("region_code") or ""
    traits = persona.get("policy_relevant_traits", {}) or {}
    active_tags = [k for k, v in traits.items() if trait_level(v) in {"중간", "높음", "낮음"}]
    issue = resolve_issue_for_100(persona, policy_keywords)
    if issue is None:
        # 이슈 미확정 — minwon/조례 등 이슈 특화 카드는 생략하고 지역 기반 카드만 생성
        return build_pack(db_path, emd, issue="__no_issue__", persona_tags=active_tags)
    return build_pack(db_path, emd, issue, active_tags)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="town_twin_full_v3_policy_stage.sqlite3")
    ap.add_argument("--emd", required=True)
    ap.add_argument("--issue", required=True)
    ap.add_argument("--persona-tags", default="")
    a = ap.parse_args()
    pack = build_pack(a.db, a.emd, a.issue, a.persona_tags.split(",") if a.persona_tags else [])
    out = f"evidence_pack_{a.emd}_{a.issue}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
    print(f"저장: {out} (카드 {len(pack['cards'])}장)")
