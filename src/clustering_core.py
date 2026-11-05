"""
화순군 100인 시민 의견 군집화 weighted 버전.

이 파일은 "정책 입력 + 100명 시민 의견 생성" 이후 단계만 담당합니다.
입력된 100명 의견을 벡터화하고, UMAP + HDBSCAN 하이퍼파라미터 탐색으로
목표 군집 수와 군집 크기 조건에 가장 잘 맞는 군집 결과를 선택합니다.

핵심 기준:
    - 목표 군집 수: 5~7개
    - 군집별 선호 인원: 14~20명
    - 평가지표: silhouette score, DBI, normalized Shannon entropy, noise ratio
    - 대표 agent: 각 군집 센트로이드에 가장 가까운 시민 1명

기본 실행:
    python 화순_군집화_weighted_0707.py

입력 파일 직접 지정:
    python 화순_군집화_weighted_0707.py --input ../hwasun_persona_policy_responses.json

no-weight 비교 실행은 별도 파일:
    python 화순_군집화_no_weight_0707.py

필수 패키지:
    python -m pip install numpy pandas openpyxl scikit-learn umap-learn hdbscan openai
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


np = None

UTF8_SIG = "utf-8-sig"
EPS = 1e-12
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_WEIGHTS = {
    "keyword": 0.40,
    "opinion": 0.45,
    "score": 0.05,
    "persona": 0.10,
}
SUPPORTED_WEIGHT_KEYS = tuple(DEFAULT_WEIGHTS.keys())


@dataclass
class CitizenResponse:
    policy_id: str
    persona_id: str
    satisfaction_score: float
    summary_keyword: str
    opinion_text: str
    region_name_kr: str = ""
    age: int | None = None
    age_group: str = ""
    persona_policy_fit: float | None = None
    sheet_name: str = ""
    raw_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterRow:
    persona_id: str
    policy_id: str
    cluster_id: int
    cluster_name: str
    is_noise: bool
    is_representative: bool
    representative_rank: int | None
    selection_reason: str
    satisfaction_score: float
    summary_keyword: str
    opinion_text: str
    region_name_kr: str
    age: int | None
    age_group: str
    persona_policy_fit: float | None
    distance_to_centroid: float | None
    cluster_size: int
    cluster_size_ratio: float
    cluster_radius: float | None
    normalized_keyword_entropy: float | None


@dataclass
class ClusterSummary:
    cluster_id: int
    cluster_name: str
    is_noise: bool
    size: int
    size_ratio: float
    avg_score: float | None
    score_distribution: dict[str, int]
    top_keywords: list[tuple[str, int]]
    top_regions: list[tuple[str, int]]
    age_groups: list[tuple[str, int]]
    keyword_entropy: float | None
    normalized_keyword_entropy: float | None
    radius: float | None
    representative_persona_id: str | None
    representative_distance: float | None
    representative_opinion: str | None


@dataclass
class SearchCandidate:
    rank: int
    objective_score: float
    selected: bool
    feature_mode: str
    cluster_count: int
    target_cluster_count_ok: bool
    target_cluster_distance: int
    preferred_size_penalty: float
    small_cluster_count: int
    large_cluster_count: int
    min_cluster_member_count: int | None
    max_cluster_member_count: int | None
    noise_ratio: float
    silhouette: float
    davies_bouldin: float
    normalized_keyword_entropy: float
    keyword_entropy: float
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    umap_n_neighbors: int
    umap_n_components: int
    umap_min_dist: float


class OptionalDependencyError(RuntimeError):
    pass


def import_numpy():
    global np
    if np is not None:
        return np
    try:
        import numpy as numpy_module
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "numpy가 설치되어 있지 않습니다.\n"
            "설치 명령: python -m pip install numpy"
        ) from exc
    np = numpy_module
    return np


def import_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "엑셀/CSV 입력을 읽으려면 pandas와 openpyxl이 필요합니다.\n"
            "설치 명령: python -m pip install pandas openpyxl"
        ) from exc
    return pd


def read_text_auto(path: Path) -> str:
    for encoding in (UTF8_SIG, "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding=UTF8_SIG, errors="replace")


def load_env_files(script_dir: Path) -> None:
    for base_dir in (script_dir, script_dir.parent):
        for env_path in (base_dir / ".env", base_dir / "env"):
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


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if 1 <= score <= 5:
            return score
    text = compact_text(value)
    match = re.search(r"([1-5])\s*점", text)
    if match:
        return float(match.group(1))
    match = re.search(r"\b([1-5](?:\.\d+)?)\b", text)
    if match:
        score = float(match.group(1))
        if 1 <= score <= 5:
            return score
    return None


def parse_age(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        age = int(value)
        if 0 < age < 120:
            return age
    text = compact_text(value)
    match = re.search(r"(\d{1,3})\s*세", text)
    if match:
        age = int(match.group(1))
        if 0 < age < 120:
            return age
    match = re.search(r"\b(\d{1,3})\b", text)
    if match:
        age = int(match.group(1))
        if 0 < age < 120:
            return age
    return None


def age_to_group(age: int | None) -> str:
    if age is None:
        return "연령미상"
    if age < 30:
        return "10-20대"
    if age < 50:
        return "30-40대"
    if age < 70:
        return "50-60대"
    return "70대이상"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if math.isnan(parsed):
            return None
        return parsed
    text = compact_text(value)
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def normalize_policy_fit(value: Any) -> float | None:
    parsed = safe_float(value)
    if parsed is None:
        return None
    if 0 <= parsed <= 1:
        return parsed
    if 1 <= parsed <= 5:
        return (parsed - 1.0) / 4.0
    if 0 <= parsed <= 100:
        return parsed / 100.0
    return max(0.0, min(1.0, parsed))


def first_present(row: dict[str, Any], aliases: Iterable[str]) -> Any:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


FIELD_ALIASES = {
    "persona_id": [
        "persona_id",
        "agent_id",
        "citizen_id",
        "id",
        "에이전트ID",
        "시민ID",
        "번호",
    ],
    "region_name_kr": ["region_name_kr", "region", "지역", "거주지역", "읍면"],
    "age": ["age", "나이", "연령"],
    "age_group": ["age_group", "연령대"],
    "satisfaction_score": [
        "satisfaction_score",
        "score",
        "rating",
        "만족도",
        "점수",
        "평점",
    ],
    "summary_keyword": [
        "summary_keyword",
        "keyword",
        "topic",
        "핵심키워드",
        "키워드",
        "쟁점",
    ],
    "opinion_text": [
        "opinion_text",
        "opinion",
        "comment",
        "text",
        "의견",
        "한줄의견",
        "한 줄 의견",
    ],
    "policy_id": ["policy_id", "policy_name", "정책명", "정책", "sheet_name"],
    "persona_policy_fit": [
        "persona_policy_fit",
        "policy_fit",
        "direct_benefit",
        "직접수혜도",
        "정책관련도",
    ],
}


EXCEL_LOG_PATTERN = re.compile(
    r"^(\d{3})\.\s+"
    r"(?P<persona_id>HW\d{3}|[A-Za-z0-9_-]+)\s+"
    r"\((?P<region>[^,()]+),\s*(?P<age>\d{1,3})\D*\)\s+"
    r"(?P<score>[1-5])\s*점\s*/\s*"
    r"(?P<keyword>[^:：]+)[:：]\s*"
    r"(?P<opinion>.+)$"
)


def parse_excel_log_cell(text: str, sheet_name: str, policy_text: str) -> CitizenResponse | None:
    match = EXCEL_LOG_PATTERN.match(text)
    if not match:
        return None
    score = parse_score(match.group("score"))
    if score is None:
        return None
    age = parse_age(match.group("age"))
    keyword = compact_text(match.group("keyword")) or "미분류"
    opinion = compact_text(match.group("opinion"))
    if not opinion:
        return None
    return CitizenResponse(
        policy_id=sheet_name or "policy",
        persona_id=match.group("persona_id"),
        region_name_kr=compact_text(match.group("region")),
        age=age,
        age_group=age_to_group(age),
        satisfaction_score=score,
        summary_keyword=keyword,
        opinion_text=opinion,
        persona_policy_fit=None,
        sheet_name=sheet_name,
        raw_text=text,
        extra={"policy_text": policy_text},
    )


def normalize_json_row(row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    for key in ["response", "policy_response", "llm_response", "result", "persona", "agent"]:
        value = row.get(key)
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                merged.setdefault(nested_key, nested_value)
    return merged


def flatten_response_container(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if not isinstance(raw, dict):
        return []
    for key in ["responses", "results", "data", "items", "records"]:
        value = raw.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [item for item in value.values() if isinstance(item, dict)]
    dict_values = [value for value in raw.values() if isinstance(value, dict)]
    if dict_values:
        return dict_values
    return [raw]


def row_to_response(row: dict[str, Any], policy_fallback: str, row_index: int) -> CitizenResponse | None:
    row = normalize_json_row(row)
    score = parse_score(first_present(row, FIELD_ALIASES["satisfaction_score"]))
    opinion = compact_text(first_present(row, FIELD_ALIASES["opinion_text"]))
    keyword = compact_text(first_present(row, FIELD_ALIASES["summary_keyword"])) or "미분류"
    if score is None or not opinion:
        return None

    persona_id = compact_text(first_present(row, FIELD_ALIASES["persona_id"])) or f"citizen_{row_index:03d}"
    age = parse_age(first_present(row, FIELD_ALIASES["age"]))
    age_group = compact_text(first_present(row, FIELD_ALIASES["age_group"])) or age_to_group(age)
    policy_id = compact_text(first_present(row, FIELD_ALIASES["policy_id"])) or policy_fallback or "policy"
    fit = normalize_policy_fit(first_present(row, FIELD_ALIASES["persona_policy_fit"]))

    return CitizenResponse(
        policy_id=policy_id,
        persona_id=persona_id,
        region_name_kr=compact_text(first_present(row, FIELD_ALIASES["region_name_kr"])),
        age=age,
        age_group=age_group,
        satisfaction_score=score,
        summary_keyword=keyword,
        opinion_text=opinion,
        persona_policy_fit=fit,
        raw_text=json.dumps(row, ensure_ascii=False),
        extra={str(key): value for key, value in row.items()},
    )


def load_from_xlsx(path: Path, sheet_name: str | None = None) -> list[CitizenResponse]:
    pd = import_pandas()
    excel = pd.ExcelFile(path)
    target_sheets = [sheet_name] if sheet_name else excel.sheet_names
    records: list[CitizenResponse] = []

    for sheet in target_sheets:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        policy_text = ""
        if raw.shape[0] > 1 and raw.shape[1] > 0:
            policy_text = compact_text(raw.iloc[1, 0])

        sheet_records: list[CitizenResponse] = []
        for _, series in raw.iterrows():
            for value in series.tolist():
                text = compact_text(value)
                if not text:
                    continue
                parsed = parse_excel_log_cell(text, sheet, policy_text)
                if parsed:
                    sheet_records.append(parsed)

        if sheet_records:
            records.extend(sheet_records)
            continue

        structured = pd.read_excel(path, sheet_name=sheet)
        structured = structured.where(pd.notna(structured), None)
        rows = structured.to_dict(orient="records")
        for idx, row in enumerate(rows, start=1):
            parsed = row_to_response(row, policy_fallback=sheet, row_index=idx)
            if parsed:
                parsed.sheet_name = sheet
                records.append(parsed)
    return records


def load_from_csv(path: Path) -> list[CitizenResponse]:
    pd = import_pandas()
    frame = pd.read_csv(path)
    frame = frame.where(pd.notna(frame), None)
    rows = frame.to_dict(orient="records")
    return [
        record
        for idx, row in enumerate(rows, start=1)
        if (record := row_to_response(row, policy_fallback=path.stem, row_index=idx))
    ]


def load_from_json(path: Path) -> list[CitizenResponse]:
    raw = json.loads(read_text_auto(path))
    policy_fallback = path.stem
    if isinstance(raw, dict) and isinstance(raw.get("policy"), dict):
        policy = raw["policy"]
        policy_fallback = compact_text(
            policy.get("policy_name")
            or policy.get("policy_id")
            or policy.get("main_details")
            or policy_fallback
        )
    rows = [normalize_json_row(row) for row in flatten_response_container(raw)]
    return [
        record
        for idx, row in enumerate(rows, start=1)
        if (record := row_to_response(row, policy_fallback=policy_fallback, row_index=idx))
    ]


def load_from_jsonl(path: Path) -> list[CitizenResponse]:
    records: list[CitizenResponse] = []
    for idx, line in enumerate(read_text_auto(path).splitlines(), start=1):
        if not line.strip():
            continue
        parsed = row_to_response(json.loads(line), policy_fallback=path.stem, row_index=idx)
        if parsed:
            records.append(parsed)
    return records


def load_responses(path: Path, sheet_name: str | None = None) -> list[CitizenResponse]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return load_from_xlsx(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return load_from_csv(path)
    if suffix == ".json":
        return load_from_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return load_from_jsonl(path)
    raise ValueError(f"지원하지 않는 입력 파일 형식입니다: {path.suffix}")


def discover_default_input(script_dir: Path) -> Path | None:
    patterns = [
        "hwasun_persona_policy_responses.json",
        "hwasun_persona_policy_responses.csv",
        "hwasun_policy_responses.json",
        "hwasun_agent_responses.json",
        "hwasun_step3_policy_responses.json",
        "responses.json",
        "responses.jsonl",
        "responses.csv",
        "responses.xlsx",
        "*TEST*.xlsx",
        "*.xlsx",
    ]
    for base_dir in (script_dir, script_dir.parent):
        for pattern in patterns:
            matches = sorted(base_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(weights.get(key, 0.0))) for key in SUPPORTED_WEIGHT_KEYS}
    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: value / total for key, value in cleaned.items()}


def parse_weights(raw: str | None) -> dict[str, float]:
    if not raw:
        return DEFAULT_WEIGHTS.copy()
    parsed = DEFAULT_WEIGHTS.copy()
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"가중치 형식이 잘못되었습니다: {chunk}")
        key, value = [part.strip() for part in chunk.split("=", 1)]
        if key not in SUPPORTED_WEIGHT_KEYS:
            raise ValueError(f"지원하지 않는 가중치 키입니다: {key}")
        parsed[key] = float(value)
    return normalize_weights(parsed)


def l2_normalize(matrix: Any) -> Any:
    import_numpy()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        script_dir: Path,
        model_name: str,
        api_key_env: str,
        batch_size: int,
    ) -> None:
        load_env_files(script_dir)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} 환경변수가 없습니다. OpenAI 임베딩은 필수이므로 여기서 중단합니다."
            )
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "openai 패키지가 설치되어 있지 않습니다.\n"
                "설치 명령: python -m pip install openai"
            ) from exc
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.cache: dict[str, list[float]] = {}

    def embed_many(self, texts: list[str]) -> Any:
        import_numpy()
        normalized_texts = [compact_text(text) or "빈 텍스트" for text in texts]
        missing: list[str] = []
        seen = set()
        for text in normalized_texts:
            if text not in self.cache and text not in seen:
                missing.append(text)
                seen.add(text)

        try:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                response = self.client.embeddings.create(model=self.model_name, input=batch)
                ordered = sorted(response.data, key=lambda item: item.index)
                for text, item in zip(batch, ordered):
                    self.cache[text] = item.embedding
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenAI Embedding API 호출 실패: {exc}") from exc

        return np.asarray([self.cache[text] for text in normalized_texts], dtype=float)


def one_hot_matrix(values: list[list[str]]) -> Any:
    import_numpy()
    try:
        from sklearn.preprocessing import OneHotEncoder
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "시민 속성 one-hot 벡터화를 위해 scikit-learn이 필요합니다.\n"
            "설치 명령: python -m pip install scikit-learn"
        ) from exc
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return np.asarray(encoder.fit_transform(values), dtype=float)


def build_embedding_text(record: CitizenResponse) -> str:
    return (
        f"핵심키워드: {record.summary_keyword or '미분류'}\n"
        f"한줄의견: {record.opinion_text or '의견없음'}"
    )


def build_weighted_features(
    records: list[CitizenResponse],
    *,
    embedder: OpenAIEmbeddingClient,
    weights: dict[str, float],
) -> tuple[Any, dict[str, Any]]:
    import_numpy()
    weights = normalize_weights(weights)
    keyword_matrix = l2_normalize(embedder.embed_many([record.summary_keyword or "미분류" for record in records]))
    opinion_matrix = l2_normalize(embedder.embed_many([record.opinion_text or "의견없음" for record in records]))
    score_matrix = np.asarray([[(record.satisfaction_score - 1.0) / 4.0] for record in records], dtype=float)
    persona_categories = [
        [record.region_name_kr or "지역미상", record.age_group or "연령미상"]
        for record in records
    ]
    persona_one_hot = one_hot_matrix(persona_categories)
    policy_fit = np.asarray(
        [[record.persona_policy_fit if record.persona_policy_fit is not None else 0.5]
         for record in records],
        dtype=float,
    )
    persona_matrix = l2_normalize(np.hstack([persona_one_hot, policy_fit]))

    blocks = {
        "keyword": keyword_matrix,
        "opinion": opinion_matrix,
        "score": score_matrix,
        "persona": persona_matrix,
    }
    weighted_blocks = [
        blocks[key] * math.sqrt(weights[key])
        for key in SUPPORTED_WEIGHT_KEYS
    ]
    features = np.hstack(weighted_blocks)
    metadata = {
        "feature_mode": "weighted_blocks",
        "keyword_backend": "openai_embedding",
        "opinion_backend": "openai_embedding",
        "openai_embedding_model": embedder.model_name,
        "weights": weights,
        "block_dimensions": {key: int(value.shape[1]) for key, value in blocks.items()},
        "block_integration": "sqrt(weight) scaling + horizontal concatenation",
        "persona_policy_fit_missing_value": 0.5,
    }
    return features, metadata


def build_embedding_only_features(
    records: list[CitizenResponse],
    *,
    embedder: OpenAIEmbeddingClient,
) -> tuple[Any, dict[str, Any]]:
    texts = [build_embedding_text(record) for record in records]
    features = l2_normalize(embedder.embed_many(texts))
    metadata = {
        "feature_mode": "embedding_only",
        "text_backend": "openai_embedding",
        "openai_embedding_model": embedder.model_name,
        "embedding_text_format": "핵심키워드 + 한줄의견",
        "block_integration": "not_used",
        "weights": "not_used",
    }
    return features, metadata


def reduce_with_umap(
    features: Any,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> Any:
    import_numpy()
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "UMAP 차원 축소를 위해 umap-learn이 필요합니다.\n"
            "설치 명령: python -m pip install umap-learn"
        ) from exc
    if len(features) < 5:
        raise ValueError("UMAP을 적용하려면 최소 5개 이상의 응답이 필요합니다.")
    output_dim = max(2, min(n_components, len(features) - 2, features.shape[1]))
    neighbor_count = max(2, min(n_neighbors, len(features) - 1))
    try:
        model = umap.UMAP(
            n_components=output_dim,
            n_neighbors=neighbor_count,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
        )
        return np.asarray(model.fit_transform(features), dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"UMAP 차원 축소 실패: {exc}") from exc


def run_hdbscan(features: Any, *, min_cluster_size: int, min_samples: int) -> Any:
    import_numpy()
    try:
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        return model.fit_predict(features)
    except ModuleNotFoundError:
        pass
    try:
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        return model.fit_predict(features)
    except (ModuleNotFoundError, ImportError, TypeError) as exc:
        raise OptionalDependencyError(
            "HDBSCAN을 사용하려면 hdbscan 패키지 또는 HDBSCAN이 포함된 scikit-learn이 필요합니다.\n"
            "설치 명령: python -m pip install hdbscan"
        ) from exc


def cluster_labels(labels: Any) -> list[int]:
    return sorted({int(label) for label in labels if int(label) != -1})


def label_sizes(labels: Any) -> dict[int, int]:
    return dict(Counter(int(label) for label in labels if int(label) != -1))


def score_distribution(scores: list[float]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for score in scores:
        key = str(int(score)) if float(score).is_integer() else str(round(float(score), 2))
        counter[key] += 1
    return dict(sorted(counter.items(), key=lambda item: float(item[0])))


def shannon_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            value -= p * math.log(p, 2)
    return value


def normalized_shannon_entropy(counter: Counter[str]) -> float:
    unique_count = len([key for key, value in counter.items() if value > 0])
    if unique_count <= 1:
        return 0.0
    return shannon_entropy(counter) / math.log(unique_count, 2)


def keyword_entropy_metrics(records: list[CitizenResponse], labels: Any) -> tuple[float, float]:
    total_non_noise = sum(1 for label in labels if int(label) != -1)
    if total_non_noise == 0:
        return 1.0, 0.0

    weighted_normalized = 0.0
    weighted_raw = 0.0
    for label in cluster_labels(labels):
        group = [record for record, current in zip(records, labels) if int(current) == label]
        counter = Counter(record.summary_keyword or "미분류" for record in group)
        weight = len(group) / total_non_noise
        weighted_normalized += weight * normalized_shannon_entropy(counter)
        weighted_raw += weight * shannon_entropy(counter)
    return float(weighted_normalized), float(weighted_raw)


def size_penalty(
    sizes: Iterable[int],
    *,
    preferred_min: int,
    preferred_max: int,
) -> tuple[float, int, int]:
    sizes = list(sizes)
    if not sizes:
        return 1.0, 0, 0
    penalties = []
    small_count = 0
    large_count = 0
    for size in sizes:
        if size < preferred_min:
            small_count += 1
            penalties.append((preferred_min - size) / max(1, preferred_min))
        elif size > preferred_max:
            large_count += 1
            penalties.append((size - preferred_max) / max(1, preferred_max))
        else:
            penalties.append(0.0)
    return float(sum(penalties) / len(penalties)), small_count, large_count


def metric_scores(records: list[CitizenResponse], features: Any, labels: Any, args: argparse.Namespace) -> dict[str, Any]:
    import_numpy()
    sizes = label_sizes(labels)
    labels_without_noise = cluster_labels(labels)
    cluster_count = len(labels_without_noise)
    noise_ratio = float(np.mean(labels == -1))
    target_distance = 0
    if cluster_count < args.target_cluster_min:
        target_distance = args.target_cluster_min - cluster_count
    elif cluster_count > args.target_cluster_max:
        target_distance = cluster_count - args.target_cluster_max

    penalty, small_count, large_count = size_penalty(
        sizes.values(),
        preferred_min=args.preferred_group_min_size,
        preferred_max=args.preferred_group_max_size,
    )
    norm_entropy, raw_entropy = keyword_entropy_metrics(records, labels)

    metrics = {
        "cluster_count": cluster_count,
        "target_cluster_count_ok": args.target_cluster_min <= cluster_count <= args.target_cluster_max,
        "target_cluster_distance": target_distance,
        "preferred_size_penalty": penalty,
        "small_cluster_count": small_count,
        "large_cluster_count": large_count,
        "min_cluster_member_count": min(sizes.values()) if sizes else None,
        "max_cluster_member_count": max(sizes.values()) if sizes else None,
        "noise_ratio": noise_ratio,
        "silhouette": -1.0,
        "davies_bouldin": 9999.0,
        "normalized_keyword_entropy": norm_entropy,
        "keyword_entropy": raw_entropy,
    }

    non_noise = labels != -1
    filtered_features = features[non_noise]
    filtered_labels = labels[non_noise]
    if cluster_count >= 2 and len(filtered_features) > cluster_count:
        try:
            from sklearn.metrics import davies_bouldin_score, silhouette_score

            metrics["silhouette"] = float(silhouette_score(filtered_features, filtered_labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(filtered_features, filtered_labels))
        except Exception:
            pass
    return metrics


def objective_score(metrics: dict[str, Any]) -> float:
    cluster_good = 1.0 / (1.0 + float(metrics["target_cluster_distance"]))
    size_good = max(0.0, 1.0 - min(1.0, float(metrics["preferred_size_penalty"])))
    silhouette_good = (float(metrics["silhouette"]) + 1.0) / 2.0
    dbi_good = 1.0 / (1.0 + max(0.0, float(metrics["davies_bouldin"])))
    entropy_good = 1.0 - min(1.0, max(0.0, float(metrics["normalized_keyword_entropy"])))
    noise_good = 1.0 - min(1.0, max(0.0, float(metrics["noise_ratio"])))
    hard_bonus = 0.05 if metrics["target_cluster_count_ok"] else 0.0
    return round(
        hard_bonus
        + 0.30 * cluster_good
        + 0.25 * size_good
        + 0.20 * silhouette_good
        + 0.10 * dbi_good
        + 0.10 * entropy_good
        + 0.05 * noise_good,
        8,
    )


def parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def hyperparameter_grid(args: argparse.Namespace) -> list[dict[str, int | float]]:
    min_cluster_sizes = parse_int_list(args.min_cluster_size_grid)
    min_samples_values = parse_int_list(args.min_samples_grid)
    n_neighbors_values = parse_int_list(args.umap_neighbors_grid)
    n_components_values = parse_int_list(args.umap_components_grid)
    rows = []
    for n_neighbors, n_components, min_cluster_size, min_samples in itertools.product(
        n_neighbors_values,
        n_components_values,
        min_cluster_sizes,
        min_samples_values,
    ):
        rows.append(
            {
                "umap_n_neighbors": n_neighbors,
                "umap_n_components": n_components,
                "umap_min_dist": args.umap_min_dist,
                "hdbscan_min_cluster_size": min_cluster_size,
                "hdbscan_min_samples": min_samples,
            }
        )
    return rows


def search_best_clustering(
    records: list[CitizenResponse],
    base_features: Any,
    *,
    args: argparse.Namespace,
    feature_mode: str,
) -> tuple[Any, Any, SearchCandidate, list[dict[str, Any]], dict[str, Any]]:
    import_numpy()
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    reduced_cache: dict[tuple[int, int, float], Any] = {}
    grid = hyperparameter_grid(args)

    for idx, params in enumerate(grid, start=1):
        cache_key = (
            int(params["umap_n_neighbors"]),
            int(params["umap_n_components"]),
            float(params["umap_min_dist"]),
        )
        if cache_key not in reduced_cache:
            reduced_cache[cache_key] = reduce_with_umap(
                base_features,
                n_components=int(params["umap_n_components"]),
                n_neighbors=int(params["umap_n_neighbors"]),
                min_dist=float(params["umap_min_dist"]),
                random_state=args.random_state,
            )
        reduced_features = reduced_cache[cache_key]
        labels = run_hdbscan(
            reduced_features,
            min_cluster_size=int(params["hdbscan_min_cluster_size"]),
            min_samples=int(params["hdbscan_min_samples"]),
        )
        metrics = metric_scores(records, reduced_features, labels, args)
        objective = objective_score(metrics)
        row = {
            "rank": 0,
            "objective_score": objective,
            "selected": False,
            "feature_mode": feature_mode,
            **metrics,
            **params,
        }
        candidates.append(row)
        if best is None or objective > best["objective_score"]:
            best = {
                "row": row,
                "labels": labels.copy(),
                "features": reduced_features.copy(),
            }
        if idx % max(1, args.progress_every) == 0:
            print(f"      탐색 진행: {idx}/{len(grid)} 조합")

    if best is None:
        raise RuntimeError("군집화 탐색 후보가 생성되지 않았습니다.")

    candidates.sort(
        key=lambda row: (
            row["objective_score"],
            1 if row["target_cluster_count_ok"] else 0,
            -row["preferred_size_penalty"],
            row["silhouette"],
            -row["davies_bouldin"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
        row["selected"] = False
    selected_key = (
        best["row"]["hdbscan_min_cluster_size"],
        best["row"]["hdbscan_min_samples"],
        best["row"]["umap_n_neighbors"],
        best["row"]["umap_n_components"],
    )
    selected_row = None
    for row in candidates:
        key = (
            row["hdbscan_min_cluster_size"],
            row["hdbscan_min_samples"],
            row["umap_n_neighbors"],
            row["umap_n_components"],
        )
        if key == selected_key and row["objective_score"] == best["row"]["objective_score"]:
            row["selected"] = True
            selected_row = row
            break
    if selected_row is None:
        selected_row = candidates[0]
        selected_row["selected"] = True

    selected_candidate = SearchCandidate(
        rank=int(selected_row["rank"]),
        objective_score=float(selected_row["objective_score"]),
        selected=True,
        feature_mode=feature_mode,
        cluster_count=int(selected_row["cluster_count"]),
        target_cluster_count_ok=bool(selected_row["target_cluster_count_ok"]),
        target_cluster_distance=int(selected_row["target_cluster_distance"]),
        preferred_size_penalty=float(selected_row["preferred_size_penalty"]),
        small_cluster_count=int(selected_row["small_cluster_count"]),
        large_cluster_count=int(selected_row["large_cluster_count"]),
        min_cluster_member_count=selected_row["min_cluster_member_count"],
        max_cluster_member_count=selected_row["max_cluster_member_count"],
        noise_ratio=float(selected_row["noise_ratio"]),
        silhouette=float(selected_row["silhouette"]),
        davies_bouldin=float(selected_row["davies_bouldin"]),
        normalized_keyword_entropy=float(selected_row["normalized_keyword_entropy"]),
        keyword_entropy=float(selected_row["keyword_entropy"]),
        hdbscan_min_cluster_size=int(selected_row["hdbscan_min_cluster_size"]),
        hdbscan_min_samples=int(selected_row["hdbscan_min_samples"]),
        umap_n_neighbors=int(selected_row["umap_n_neighbors"]),
        umap_n_components=int(selected_row["umap_n_components"]),
        umap_min_dist=float(selected_row["umap_min_dist"]),
    )

    # selected_row가 정렬 뒤 첫 행이 아닐 수 있으므로 최종 파라미터로 다시 재현한다.
    selected_features = reduce_with_umap(
        base_features,
        n_components=selected_candidate.umap_n_components,
        n_neighbors=selected_candidate.umap_n_neighbors,
        min_dist=selected_candidate.umap_min_dist,
        random_state=args.random_state,
    )
    selected_labels = run_hdbscan(
        selected_features,
        min_cluster_size=selected_candidate.hdbscan_min_cluster_size,
        min_samples=selected_candidate.hdbscan_min_samples,
    )
    reducer_metadata = {
        "reducer": "umap",
        "input_dim": int(base_features.shape[1]),
        "output_dim": int(selected_features.shape[1]),
        "n_neighbors": selected_candidate.umap_n_neighbors,
        "n_components": selected_candidate.umap_n_components,
        "min_dist": selected_candidate.umap_min_dist,
        "metric": "cosine",
        "searched_combinations": len(grid),
        "cached_umap_runs": len(reduced_cache),
    }
    return selected_features, selected_labels, selected_candidate, candidates, reducer_metadata


def cluster_name(records: list[CitizenResponse], label: int) -> str:
    if label == -1:
        return "노이즈/개별의견"
    keywords = [
        record.summary_keyword
        for record in records
        if record.summary_keyword and record.summary_keyword != "미분류"
    ]
    top_keywords = [keyword for keyword, _ in Counter(keywords).most_common(2)]
    if not top_keywords:
        return "의견 쟁점군"
    if len(top_keywords) == 1:
        return f"{top_keywords[0]} 쟁점군"
    return f"{top_keywords[0]}/{top_keywords[1]} 쟁점군"


def centroids_and_distances(features: Any, labels: Any) -> tuple[dict[int, Any], list[float | None]]:
    import_numpy()
    centroids: dict[int, Any] = {}
    for label in cluster_labels(labels):
        members = features[labels == label]
        centroids[label] = members.mean(axis=0)
    distances: list[float | None] = []
    for row, label in zip(features, labels):
        label = int(label)
        if label == -1 or label not in centroids:
            distances.append(None)
        else:
            distances.append(float(np.linalg.norm(row - centroids[label])))
    return centroids, distances


def representative_ranks(labels: Any, distances: list[float | None]) -> dict[int, int]:
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for idx, (label, distance) in enumerate(zip(labels, distances)):
        label = int(label)
        if label == -1 or distance is None:
            continue
        grouped[label].append((idx, distance))
    ranks: dict[int, int] = {}
    for _, values in grouped.items():
        for rank, (idx, _) in enumerate(sorted(values, key=lambda item: item[1]), start=1):
            ranks[idx] = rank
    return ranks


def build_outputs(
    records: list[CitizenResponse],
    features: Any,
    labels: Any,
) -> tuple[list[ClusterRow], list[ClusterSummary], list[dict[str, Any]]]:
    import_numpy()
    _, distances = centroids_and_distances(features, labels)
    ranks = representative_ranks(labels, distances)
    label_to_indexes: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indexes[int(label)].append(idx)

    label_summaries: dict[int, ClusterSummary] = {}
    representative_indexes: set[int] = set()

    for label in sorted(label_to_indexes):
        indexes = label_to_indexes[label]
        group = [records[idx] for idx in indexes]
        scores = [record.satisfaction_score for record in group]
        keywords = Counter(record.summary_keyword or "미분류" for record in group)
        regions = Counter(record.region_name_kr or "지역미상" for record in group)
        ages = Counter(record.age_group or "연령미상" for record in group)
        is_noise = label == -1
        radius = None
        representative_idx = None
        representative_distance = None
        if not is_noise:
            member_distances = [distances[idx] for idx in indexes if distances[idx] is not None]
            radius = float(np.mean(member_distances)) if member_distances else None
            representative_idx = min(
                indexes,
                key=lambda idx: distances[idx] if distances[idx] is not None else float("inf"),
            )
            representative_indexes.add(representative_idx)
            representative_distance = distances[representative_idx]

        label_summaries[label] = ClusterSummary(
            cluster_id=label,
            cluster_name=cluster_name(group, label),
            is_noise=is_noise,
            size=len(indexes),
            size_ratio=round(len(indexes) / len(records), 4),
            avg_score=round(float(np.mean(scores)), 4) if scores else None,
            score_distribution=score_distribution(scores),
            top_keywords=keywords.most_common(8),
            top_regions=regions.most_common(5),
            age_groups=ages.most_common(),
            keyword_entropy=round(shannon_entropy(keywords), 6) if not is_noise else None,
            normalized_keyword_entropy=round(normalized_shannon_entropy(keywords), 6) if not is_noise else None,
            radius=round(radius, 6) if radius is not None else None,
            representative_persona_id=records[representative_idx].persona_id if representative_idx is not None else None,
            representative_distance=round(representative_distance, 6) if representative_distance is not None else None,
            representative_opinion=records[representative_idx].opinion_text if representative_idx is not None else None,
        )

    rows: list[ClusterRow] = []
    for idx, (record, label_value) in enumerate(zip(records, labels)):
        label = int(label_value)
        summary = label_summaries[label]
        is_representative = idx in representative_indexes
        rows.append(
            ClusterRow(
                persona_id=record.persona_id,
                policy_id=record.policy_id,
                cluster_id=label,
                cluster_name=summary.cluster_name,
                is_noise=summary.is_noise,
                is_representative=is_representative,
                representative_rank=ranks.get(idx),
                selection_reason="군집 센트로이드 최단거리 대표" if is_representative else "",
                satisfaction_score=record.satisfaction_score,
                summary_keyword=record.summary_keyword,
                opinion_text=record.opinion_text,
                region_name_kr=record.region_name_kr,
                age=record.age,
                age_group=record.age_group,
                persona_policy_fit=record.persona_policy_fit,
                distance_to_centroid=round(distances[idx], 6) if distances[idx] is not None else None,
                cluster_size=summary.size,
                cluster_size_ratio=summary.size_ratio,
                cluster_radius=summary.radius,
                normalized_keyword_entropy=summary.normalized_keyword_entropy,
            )
        )

    summaries = [label_summaries[label] for label in sorted(label_summaries)]
    representatives = []
    for row in rows:
        if not row.is_representative or row.is_noise:
            continue
        payload = asdict(row)
        payload["selection_reason"] = "해당 군집의 센트로이드와 가장 가까운 시민"
        payload["representative_selection_method"] = "nearest_to_cluster_centroid"
        representatives.append(payload)
    representatives.sort(key=lambda item: (item["cluster_id"], item["persona_id"]))
    return rows, summaries, representatives


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def csv_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_ready(value), ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding=UTF8_SIG)
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding=UTF8_SIG, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key)) for key in fieldnames})


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2), encoding=UTF8_SIG)


def make_output_dir(base_dir: Path, requested: str | None, feature_mode: str) -> Path:
    if requested:
        output_dir = Path(requested).expanduser()
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / "outputs" / f"hwasun_clustering_{feature_mode}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_arg_parser(default_feature_mode: str = "weighted_blocks") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="화순군 100인 시민 의견 군집화 weighted/no-weight 공통 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None, help="응답 입력 파일 경로(JSON/JSONL/CSV/XLSX)")
    parser.add_argument("--sheet", type=str, default=None, help="엑셀에서 특정 시트만 분석할 때 지정")
    parser.add_argument("--output-dir", type=str, default=None, help="결과 저장 폴더")
    parser.add_argument(
        "--feature-mode",
        choices=["weighted_blocks", "embedding_only"],
        default=default_feature_mode,
        help="weighted_blocks는 키워드/의견/만족도/페르소나 가중 통합, embedding_only는 API 임베딩만 사용",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="keyword=0.40,opinion=0.45,score=0.05,persona=0.10",
        help="weighted_blocks 모드 가중치",
    )
    parser.add_argument("--openai-embedding-model", type=str, default=DEFAULT_OPENAI_EMBEDDING_MODEL)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--embedding-batch-size", type=int, default=100)
    parser.add_argument("--target-cluster-min", type=int, default=5)
    parser.add_argument("--target-cluster-max", type=int, default=7)
    parser.add_argument("--preferred-group-min-size", type=int, default=14)
    parser.add_argument("--preferred-group-max-size", type=int, default=20)
    parser.add_argument(
        "--min-cluster-size-grid",
        type=str,
        default="5,8,10,12,14,16,18,20",
        help="HDBSCAN min_cluster_size 탐색 후보",
    )
    parser.add_argument(
        "--min-samples-grid",
        type=str,
        default="1,2,3,5,8",
        help="HDBSCAN min_samples 탐색 후보",
    )
    parser.add_argument(
        "--umap-neighbors-grid",
        type=str,
        default="8,10,12,15,20",
        help="UMAP n_neighbors 탐색 후보",
    )
    parser.add_argument(
        "--umap-components-grid",
        type=str,
        default="5,8,10",
        help="UMAP n_components 탐색 후보",
    )
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50, help="탐색 진행 상황 출력 간격")
    return parser


def run_pipeline(args: argparse.Namespace) -> Path:
    import_numpy()
    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input).expanduser() if args.input else discover_default_input(script_dir)
    if input_path is None:
        raise SystemExit(
            "입력 파일을 찾지 못했습니다. --input 경로를 지정하거나 TT_3 또는 부모 TT 폴더에 responses 파일을 두세요."
        )
    if not input_path.is_absolute():
        input_path = (script_dir / input_path).resolve()
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")

    output_dir = make_output_dir(script_dir, args.output_dir, args.feature_mode)
    print(f"[1/7] 시민 응답 읽는 중: {input_path}")
    records = load_responses(input_path, sheet_name=args.sheet)
    if not records:
        raise SystemExit("파싱된 시민 응답이 없습니다. 입력 파일의 만족도/키워드/의견 컬럼을 확인하세요.")
    print(f"      파싱 완료: {len(records)}건")

    print("[2/7] OpenAI 임베딩 준비 중")
    embedder = OpenAIEmbeddingClient(
        script_dir=script_dir,
        model_name=args.openai_embedding_model,
        api_key_env=args.api_key_env,
        batch_size=args.embedding_batch_size,
    )

    print(f"[3/7] 피처 생성 중: {args.feature_mode}")
    if args.feature_mode == "weighted_blocks":
        base_features, feature_metadata = build_weighted_features(
            records,
            embedder=embedder,
            weights=parse_weights(args.weights),
        )
    else:
        base_features, feature_metadata = build_embedding_only_features(records, embedder=embedder)

    grid_count = len(hyperparameter_grid(args))
    print(f"[4/7] UMAP + HDBSCAN 하이퍼파라미터 탐색 중: {grid_count}개 조합")
    selected_features, selected_labels, selected_candidate, search_rows, reducer_metadata = search_best_clustering(
        records,
        base_features,
        args=args,
        feature_mode=args.feature_mode,
    )

    print("[5/7] 군집 해석 및 대표 agent 선정 중")
    clustered_rows, cluster_summaries, representatives = build_outputs(records, selected_features, selected_labels)

    print("[6/7] 결과 저장 중")
    write_csv(output_dir / "clustered_responses.csv", [asdict(row) for row in clustered_rows])
    write_json(output_dir / "cluster_summary.json", [asdict(summary) for summary in cluster_summaries])
    write_json(output_dir / "representative_agents.json", representatives)
    write_csv(output_dir / "hyperparameter_search_results.csv", search_rows)

    run_summary = {
        "input_path": str(input_path),
        "sheet": args.sheet,
        "record_count": len(records),
        "feature_mode": args.feature_mode,
        "algorithm": "umap+hdbscan",
        "target_cluster_count": {
            "min": args.target_cluster_min,
            "max": args.target_cluster_max,
        },
        "preferred_group_size": {
            "min": args.preferred_group_min_size,
            "max": args.preferred_group_max_size,
            "reason": "100명 기준 목표 군집 수 5~7개를 유도하기 위한 군집당 평균 인원 범위",
        },
        "selected_candidate": asdict(selected_candidate),
        "feature_metadata": feature_metadata,
        "reducer_metadata": reducer_metadata,
        "evaluation_metrics": {
            "silhouette": selected_candidate.silhouette,
            "davies_bouldin": selected_candidate.davies_bouldin,
            "normalized_keyword_entropy": selected_candidate.normalized_keyword_entropy,
            "keyword_entropy": selected_candidate.keyword_entropy,
            "noise_ratio": selected_candidate.noise_ratio,
            "preferred_size_penalty": selected_candidate.preferred_size_penalty,
        },
        "selection_logic": {
            "objective_score": (
                "0.30*cluster_count_fit + 0.25*group_size_fit + "
                "0.20*silhouette + 0.10*DBI_fit + 0.10*entropy_fit + 0.05*noise_fit + target_bonus"
            ),
            "representative_selection": "각 비노이즈 군집의 센트로이드 최단거리 시민 1명",
        },
        "technology_pipeline": {
            "citizen_response_generation": "external_team_code",
            "feature_generation": args.feature_mode,
            "dimension_reduction": "UMAP",
            "clustering": "HDBSCAN",
            "cluster_evaluation": "silhouette, DBI, normalized Shannon entropy, noise ratio, group size penalty",
            "representative_selection": "nearest_to_cluster_centroid",
        },
        "output_files": [
            "clustered_responses.csv",
            "cluster_summary.json",
            "representative_agents.json",
            "hyperparameter_search_results.csv",
            "run_summary.json",
        ],
    }
    write_json(output_dir / "run_summary.json", run_summary)

    print("[7/7] 완료")
    print(f"- 결과 폴더: {output_dir}")
    print(f"- 선택 군집 수: {selected_candidate.cluster_count}개")
    print(f"- 대표 agent 수: {len(representatives)}명")
    print(f"- silhouette: {selected_candidate.silhouette:.4f}")
    print(f"- DBI: {selected_candidate.davies_bouldin:.4f}")
    print(f"- normalized entropy: {selected_candidate.normalized_keyword_entropy:.4f}")
    print(f"- noise ratio: {selected_candidate.noise_ratio:.2%}")
    if not selected_candidate.target_cluster_count_ok:
        print("[주의] 목표 군집 수 5~7개를 만족하지 못했습니다. search_results를 확인하세요.")
    return output_dir


def run_cli(default_feature_mode: str = "weighted_blocks") -> None:
    args = build_arg_parser(default_feature_mode=default_feature_mode).parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    try:
        run_cli(default_feature_mode="weighted_blocks")
    except OptionalDependencyError as exc:
        raise SystemExit(str(exc)) from exc
