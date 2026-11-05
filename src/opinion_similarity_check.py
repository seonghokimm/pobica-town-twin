#!/usr/bin/env python3
"""Compare generated policy reactions with real public-opinion text.

The script reports two complementary signals:
- keyword-vector cosine over an explicit keyword dictionary
- TF-IDF cosine between generated opinions and real opinion documents
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_SQLITE = Path("town_twin_full_v3_policy_stage.sqlite3")


@dataclass
class TextRecord:
    record_id: str
    text: str
    keyword: str = ""
    source: str = ""
    match_terms: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify similarity between generated responses and real public opinion."
    )
    parser.add_argument("--responses", help="Generated response JSON or CSV")
    parser.add_argument(
        "--cluster-dir",
        help="Optional clustering output dir. clustered_responses.csv is preferred when present.",
    )
    parser.add_argument("--ai-debate-result", help="Optional debate_result.json used as AI opinion/issues")
    parser.add_argument(
        "--debate-scope",
        choices=["full", "discussion_only"],
        default="full",
        help=(
            "Debate extraction scope. full includes representatives and cluster keywords; "
            "discussion_only uses only questions, mediations, final proposals, and moderator summary."
        ),
    )
    parser.add_argument(
        "--ai-source",
        choices=["generated", "debate", "both"],
        default="generated",
        help="AI-side comparison source: generated responses, debate output, or both.",
    )
    parser.add_argument("--actual-csv", help="CSV with real public-opinion text")
    parser.add_argument(
        "--actual-urls-file",
        help="Text file with one news/article URL per line. Lines may optionally be issue_label<TAB>url.",
    )
    parser.add_argument(
        "--reference-report-data",
        help=(
            "Debate report_data.json used as reference issues. "
            "This checks debate coverage of generated responses, not real-world opinion validity."
        ),
    )
    parser.add_argument("--actual-text-col", default="content", help="Text column in --actual-csv")
    parser.add_argument("--actual-id-col", default="", help="Optional ID column in --actual-csv")
    parser.add_argument(
        "--actual-keyword-col",
        default="",
        help="Optional issue/keyword column in --actual-csv, e.g. 실제 주민 의견",
    )
    parser.add_argument(
        "--actual-match-terms-col",
        default="match_terms",
        help=(
            "Optional coverage term column in --actual-csv. "
            "Use semicolons for required groups and / for alternatives, e.g. 정산/환전;호출/배차."
        ),
    )
    parser.add_argument("--sqlite", default=str(DEFAULT_SQLITE), help="SQLite RAG DB for real text")
    parser.add_argument(
        "--terms",
        default="",
        help="Comma-separated terms used to filter SQLite real-opinion documents",
    )
    parser.add_argument(
        "--keywords",
        default="",
        help="Comma-separated controlled keywords for keyword cosine. Defaults to terms + generated summary keywords.",
    )
    parser.add_argument("--limit", type=int, default=300, help="Max real documents from SQLite")
    parser.add_argument("--output-dir", default="", help="Directory for report files")
    parser.add_argument(
        "--crr-threshold",
        type=float,
        default=0.75,
        help="Pair cosine threshold counted as represented in CRR.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.5,
        help="Issue term coverage threshold counted as represented.",
    )
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean_text(value: object) -> str:
    return str(value or "").replace("\ufeff", "").strip()


def clean_keyword_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return clean_text(value[0] if value else "")
    if isinstance(value, dict):
        for key in ["keyword", "name", "label", "쟁점"]:
            if key in value:
                return clean_text(value[key])
    return clean_text(value)


def split_csv_terms(raw: str) -> list[str]:
    terms: list[str] = []
    for part in raw.split(","):
        term = part.strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def load_generated(responses: Path, cluster_dir: Path | None) -> list[TextRecord]:
    clustered = cluster_dir / "clustered_responses.csv" if cluster_dir else None
    if clustered and clustered.exists():
        df = pd.read_csv(clustered, encoding="utf-8-sig")
        records = []
        for idx, row in df.iterrows():
            text = clean_text(row.get("opinion_text"))
            if not text:
                continue
            records.append(
                TextRecord(
                    record_id=clean_text(row.get("persona_id")) or f"generated-{idx + 1}",
                    text=text,
                    keyword=clean_text(row.get("summary_keyword")),
                    source="generated_clustered",
                )
            )
        return records

    if responses.suffix.lower() == ".csv":
        df = pd.read_csv(responses, encoding="utf-8-sig")
        records = []
        for idx, row in df.iterrows():
            text = clean_text(row.get("opinion_text") or row.get("one_line_opinion"))
            if not text:
                continue
            records.append(
                TextRecord(
                    record_id=clean_text(row.get("persona_id")) or f"generated-{idx + 1}",
                    text=text,
                    keyword=clean_text(row.get("summary_keyword") or row.get("key_keywords")),
                    source="generated",
                )
            )
        return records

    data = read_json(responses)
    rows = data.get("results", data) if isinstance(data, dict) else data
    records = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = clean_text(row.get("opinion_text") or row.get("one_line_opinion"))
        if not text:
            continue
        keywords = row.get("key_keywords")
        if isinstance(keywords, list):
            keyword = ", ".join(clean_text(k) for k in keywords if clean_text(k))
        else:
            keyword = clean_text(row.get("summary_keyword"))
        records.append(
            TextRecord(
                record_id=clean_text(row.get("persona_id")) or f"generated-{idx + 1}",
                text=text,
                keyword=keyword,
                source="generated",
            )
        )
    return records


def load_generated_from_debate(path: Path, scope: str = "full") -> list[TextRecord]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a debate_result.json object")

    records: list[TextRecord] = []

    def add(record_id: str, keyword: object, text: object, source: str) -> None:
        label = clean_keyword_value(keyword)
        body = clean_text(text)
        if not label and not body:
            return
        records.append(
            TextRecord(
                record_id=record_id,
                text=body or label,
                keyword=label or body[:80],
                source=source,
            )
        )

    if scope == "full":
        for idx, rep in enumerate(data.get("representatives") or [], start=1):
            if isinstance(rep, dict):
                add(
                    f"debate-representative-{idx}",
                    rep.get("summary_keyword"),
                    rep.get("opinion_text"),
                    "debate.representatives",
                )

        for idx, cluster in enumerate(data.get("cluster_summary") or [], start=1):
            if not isinstance(cluster, dict):
                continue
            for kw_idx, keyword in enumerate(cluster.get("top_keywords") or [], start=1):
                add(
                    f"debate-cluster-{idx}-keyword-{kw_idx}",
                    keyword,
                    cluster.get("representative_opinion"),
                    "debate.cluster_summary.top_keywords",
                )

    debate = data.get("debate") or {}
    if scope == "full":
        for idx, opening in enumerate(debate.get("openings") or [], start=1):
            if isinstance(opening, dict):
                add(
                    f"debate-opening-{idx}",
                    opening.get("claim"),
                    opening.get("핵심근거") or opening.get("warrant"),
                    "debate.openings",
                )

    for idx, question in enumerate(debate.get("questions") or [], start=1):
        if isinstance(question, dict):
            add(
                f"debate-question-{idx}",
                question.get("쟁점") or question.get("claim"),
                question.get("질문") or question.get("warrant"),
                "debate.questions",
            )

    for idx, mediation in enumerate(debate.get("mediations") or [], start=1):
        if not isinstance(mediation, dict):
            continue
        responses = mediation.get("대표응답") or []
        response_text = " ".join(clean_text(item) for item in responses if clean_text(item))
        add(
            f"debate-mediation-{idx}",
            mediation.get("쟁점"),
            " ".join(
                part
                for part in [
                    clean_text(mediation.get("사회자질문")),
                    response_text,
                    clean_text(mediation.get("조정정리")),
                ]
                if part
            ),
            "debate.mediations",
        )

    for idx, proposal in enumerate(debate.get("final_proposals") or [], start=1):
        if isinstance(proposal, dict):
            add(
                f"debate-final-proposal-{idx}",
                proposal.get("키워드"),
                " ".join(
                    part
                    for part in [
                        clean_text(proposal.get("최종입장")),
                        clean_text(proposal.get("핵심요구")),
                        clean_text(proposal.get("양보가능지점")),
                    ]
                    if part
                ),
                "debate.final_proposals",
            )

    summary = debate.get("moderator_summary") or {}
    if isinstance(summary, dict):
        for field in ["핵심쟁점", "주민보완요구", "미해결쟁점"]:
            values = summary.get(field) or []
            if isinstance(values, str):
                values = [values]
            for idx, value in enumerate(values, start=1):
                add(f"debate-summary-{field}-{idx}", value, value, f"debate.moderator_summary.{field}")

    return records


def load_actual_from_csv(
    path: Path, text_col: str, id_col: str, keyword_col: str, match_terms_col: str
) -> list[TextRecord]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if text_col not in df.columns:
        raise ValueError(f"{path} has no text column: {text_col}")
    records = []
    for idx, row in df.iterrows():
        text = clean_text(row.get(text_col))
        if not text:
            continue
        record_id = clean_text(row.get(id_col)) if id_col and id_col in df.columns else f"actual-{idx + 1}"
        keyword = clean_text(row.get(keyword_col)) if keyword_col and keyword_col in df.columns else ""
        match_terms = clean_text(row.get(match_terms_col)) if match_terms_col and match_terms_col in df.columns else ""
        records.append(
            TextRecord(
                record_id=record_id,
                text=text,
                keyword=keyword,
                source="actual_csv",
                match_terms=match_terms,
            )
        )
    return records


def extract_article_text(url: str) -> tuple[str, str]:
    response = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/126.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form"]):
        tag.decompose()

    title = clean_text(soup.find("meta", property="og:title", content=True) and soup.find("meta", property="og:title")["content"])
    if not title and soup.title:
        title = clean_text(soup.title.get_text(" "))

    candidates = []
    selectors = [
        "article",
        "[itemprop='articleBody']",
        ".article_body",
        ".article-body",
        ".news_body",
        ".news-body",
        "#articleBody",
        "#article_body",
        "#newsContent",
        "#article-view-content-div",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            text = normalize_article_text(node.get_text("\n"))
            if len(text) >= 200:
                candidates.append(text)

    if not candidates:
        paragraphs = [normalize_article_text(p.get_text(" ")) for p in soup.find_all("p")]
        candidates = ["\n".join(p for p in paragraphs if len(p) >= 20)]

    text = max(candidates, key=len, default="")
    return title, text


def normalize_article_text(text: str) -> str:
    text = re.sub(r"\s+", " ", clean_text(text))
    boilerplate = [
        "무단전재",
        "재배포 금지",
        "저작권자",
        "Copyright",
        "All rights reserved",
    ]
    for marker in boilerplate:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text.strip()


def load_actual_from_urls_file(path: Path) -> list[TextRecord]:
    records: list[TextRecord] = []
    for idx, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            keyword, url = [part.strip() for part in line.split("\t", 1)]
        else:
            keyword, url = "", line
        try:
            title, text = extract_article_text(url)
        except Exception as exc:
            print(f"[warn] URL fetch failed: {url} ({exc})")
            continue
        if not text:
            print(f"[warn] No article text extracted: {url}")
            continue
        records.append(
            TextRecord(
                record_id=f"url-{idx}",
                text=text,
                keyword=keyword or title,
                source=url,
            )
        )
    return records


def load_reference_from_report_data(path: Path) -> list[TextRecord]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a report_data-style JSON object")

    records: list[TextRecord] = []

    def add(label: str, text: object, source: str) -> None:
        issue = clean_text(label)
        body = clean_text(text)
        if not issue and not body:
            return
        records.append(
            TextRecord(
                record_id=f"report-{len(records) + 1}",
                text=body or issue,
                keyword=issue or body[:80],
                source=source,
            )
        )

    for keyword in data.get("top_keywords") or []:
        add(clean_text(keyword), clean_text(keyword), "report_data.top_keywords")

    for issue in data.get("issues") or []:
        add(clean_text(issue), clean_text(issue), "report_data.issues")

    discussion_summary = clean_text(data.get("discussion_summary"))
    for part in re.split(r"[,;/·]\s*", discussion_summary):
        part = clean_text(part)
        if part:
            add(part, part, "report_data.discussion_summary")

    for idx, consensus in enumerate(data.get("consensus") or [], start=1):
        text = clean_text(consensus)
        if text:
            add(text[:80], text, f"report_data.consensus.{idx}")

    for idx, item in enumerate(data.get("improvements") or [], start=1):
        if isinstance(item, dict):
            title = clean_text(item.get("title"))
            reason = clean_text(item.get("reason"))
            add(title[:80], " ".join(part for part in [title, reason] if part), f"report_data.improvements.{idx}")
        else:
            text = clean_text(item)
            add(text[:80], text, f"report_data.improvements.{idx}")

    return records


def load_actual_from_sqlite(path: Path, terms: list[str], limit: int) -> list[TextRecord]:
    if not terms:
        raise ValueError("--terms is required when using --sqlite")
    con = sqlite3.connect(path)
    where = " OR ".join(["search_text LIKE ?" for _ in terms])
    params = [f"%{term}%" for term in terms]
    sql = f"""
        SELECT rag_id, source_type, title, content
        FROM rag_documents
        WHERE {where}
        ORDER BY
          CASE WHEN source_type = 'minwon' THEN 0 ELSE 1 END,
          rag_id
        LIMIT ?
    """
    records = []
    for rag_id, source_type, title, content in con.execute(sql, [*params, limit]):
        text = clean_text(content)
        if text:
            keyword = next((term for term in terms if term in text), clean_text(title))
            records.append(
                TextRecord(
                    record_id=clean_text(rag_id),
                    text=text,
                    keyword=keyword,
                    source=f"{clean_text(source_type)}:{clean_text(title)}",
                )
            )
    return records


def count_keyword_vector(records: Iterable[TextRecord], keywords: list[str]) -> np.ndarray:
    counts = []
    texts = [record.text for record in records]
    joined = "\n".join(texts)
    for keyword in keywords:
        counts.append(joined.count(keyword))
    return np.asarray(counts, dtype=float)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def tfidf_metrics(generated: list[TextRecord], actual: list[TextRecord]) -> dict[str, float]:
    texts = [r.text for r in generated] + [r.text for r in actual]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    gen_matrix = matrix[: len(generated)]
    actual_matrix = matrix[len(generated) :]

    gen_centroid = np.asarray(gen_matrix.mean(axis=0))
    actual_centroid = np.asarray(actual_matrix.mean(axis=0))
    centroid_similarity = cosine_similarity(gen_centroid, actual_centroid)[0, 0]
    pairwise = cosine_similarity(actual_matrix, gen_matrix)
    actual_best = pairwise.max(axis=1)
    generated_best = pairwise.max(axis=0)
    return {
        "tfidf_centroid_cosine": float(centroid_similarity),
        "actual_to_generated_best_mean": float(mean(actual_best)),
        "actual_to_generated_best_median": float(median(actual_best)),
        "actual_to_generated_best_p90": float(np.quantile(actual_best, 0.9)),
        "generated_to_actual_best_mean": float(mean(generated_best)),
    }


def unique_issue_records(records: list[TextRecord], fallback_prefix: str) -> list[TextRecord]:
    seen: set[str] = set()
    issues: list[TextRecord] = []
    for idx, record in enumerate(records):
        issue = clean_text(record.keyword) or clean_text(record.text)
        if not issue or issue in seen:
            continue
        seen.add(issue)
        compare_text = " ".join(
            part for part in [clean_text(record.keyword), clean_text(record.text)] if part
        )
        issues.append(
            TextRecord(
                record_id=record.record_id or f"{fallback_prefix}-{idx + 1}",
                text=compare_text or issue,
                keyword=issue,
                source=record.source,
            )
        )
    return issues


def crr_pair_table(
    actual: list[TextRecord], generated: list[TextRecord], threshold: float
) -> tuple[dict[str, float], list[dict[str, object]]]:
    actual_issues = unique_issue_records(actual, "actual-issue")
    generated_issues = unique_issue_records(generated, "generated-issue")
    if not actual_issues or not generated_issues:
        return {
            "crr_mean_best_cosine": 0.0,
            "crr_percent": 0.0,
            "crr_represented_share": 0.0,
            "crr_threshold": threshold,
        }, []

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform([r.text for r in actual_issues] + [r.text for r in generated_issues])
    actual_matrix = matrix[: len(actual_issues)]
    generated_matrix = matrix[len(actual_issues) :]
    pairwise = cosine_similarity(actual_matrix, generated_matrix)

    rows: list[dict[str, object]] = []
    best_scores: list[float] = []
    for actual_idx, actual_issue in enumerate(actual_issues):
        best_idx = int(pairwise[actual_idx].argmax())
        score = float(pairwise[actual_idx, best_idx])
        best_scores.append(score)
        generated_issue = generated_issues[best_idx]
        rows.append(
            {
                "actual_opinion": actual_issue.keyword,
                "ai_opinion": generated_issue.keyword,
                "cosine_similarity": round(score, 4),
                "represented": score >= threshold,
                "actual_id": actual_issue.record_id,
                "ai_id": generated_issue.record_id,
            }
        )

    represented_share = float(sum(score >= threshold for score in best_scores) / len(best_scores))
    mean_best = float(mean(best_scores))
    return {
        "crr_mean_best_cosine": mean_best,
        "crr_percent": mean_best * 100.0,
        "crr_represented_share": represented_share,
        "crr_threshold": threshold,
    }, sorted(rows, key=lambda row: float(row["cosine_similarity"]), reverse=True)


def parse_match_term_groups(raw: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for group in re.split(r"\s*;\s*", clean_text(raw)):
        alternatives = [term.strip() for term in re.split(r"\s*[/|]\s*", group) if term.strip()]
        if alternatives:
            groups.append(alternatives)
    return groups


def term_coverage_table(
    actual: list[TextRecord], generated: list[TextRecord], threshold: float
) -> tuple[dict[str, float], list[dict[str, object]]]:
    generated_texts = [
        " ".join(part for part in [record.keyword, record.text] if clean_text(part))
        for record in generated
    ]
    rows: list[dict[str, object]] = []
    scores: list[float] = []

    for actual_record in actual:
        groups = parse_match_term_groups(actual_record.match_terms)
        if not groups:
            continue

        best: dict[str, object] | None = None
        for generated_record, generated_text in zip(generated, generated_texts):
            matched_groups: list[str] = []
            missing_groups: list[str] = []
            for alternatives in groups:
                matched = next((term for term in alternatives if term and term in generated_text), "")
                if matched:
                    matched_groups.append(matched)
                else:
                    missing_groups.append("/".join(alternatives))

            score = len(matched_groups) / len(groups)
            row = {
                "actual_opinion": actual_record.keyword or actual_record.record_id,
                "ai_opinion": generated_record.keyword,
                "coverage_score": round(score, 4),
                "coverage_percent": round(score * 100.0, 2),
                "represented": score >= threshold,
                "matched_terms": ", ".join(matched_groups),
                "missing_terms": ", ".join(missing_groups),
                "actual_id": actual_record.record_id,
                "ai_id": generated_record.record_id,
            }
            if best is None or (score, len(row["matched_terms"])) > (
                float(best["coverage_score"]),
                len(str(best["matched_terms"])),
            ):
                best = row

        if best is not None:
            scores.append(float(best["coverage_score"]))
            rows.append(best)

    if not scores:
        return {
            "issue_coverage_mean": 0.0,
            "issue_coverage_percent": 0.0,
            "issue_coverage_represented_share": 0.0,
            "issue_coverage_threshold": threshold,
        }, []

    represented_share = sum(score >= threshold for score in scores) / len(scores)
    mean_score = float(mean(scores))
    return {
        "issue_coverage_mean": mean_score,
        "issue_coverage_percent": mean_score * 100.0,
        "issue_coverage_represented_share": float(represented_share),
        "issue_coverage_threshold": threshold,
    }, sorted(rows, key=lambda row: float(row["coverage_score"]), reverse=True)


def write_keyword_csv(path: Path, keywords: list[str], gen_vec: np.ndarray, actual_vec: np.ndarray) -> None:
    gen_total = float(gen_vec.sum()) or 1.0
    actual_total = float(actual_vec.sum()) or 1.0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keyword", "generated_count", "actual_count", "generated_share", "actual_share"])
        for keyword, gen_count, actual_count in sorted(
            zip(keywords, gen_vec, actual_vec), key=lambda row: (row[1] + row[2], row[2]), reverse=True
        ):
            writer.writerow(
                [
                    keyword,
                    int(gen_count),
                    int(actual_count),
                    round(float(gen_count) / gen_total, 6),
                    round(float(actual_count) / actual_total, 6),
                ]
            )


def write_pair_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "actual_opinion",
                "ai_opinion",
                "cosine_similarity",
                "represented",
                "actual_id",
                "ai_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_coverage_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "actual_opinion",
                "ai_opinion",
                "coverage_score",
                "coverage_percent",
                "represented",
                "matched_terms",
                "missing_terms",
                "actual_id",
                "ai_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_actual_sources_csv(path: Path, records: list[TextRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual_id", "issue_label", "source", "text_preview"])
        for record in records:
            writer.writerow([record.record_id, record.keyword, record.source, record.text[:300]])


def write_report(
    path: Path,
    metrics: dict[str, float],
    generated: list[TextRecord],
    actual: list[TextRecord],
    keywords: list[str],
    pair_rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
) -> None:
    keyword_counter = Counter(r.keyword for r in generated if r.keyword)
    lines = [
        "# 실제 여론 유사도 정량 검증",
        "",
        f"- 가상 응답 수: {len(generated)}",
        f"- 실제 여론 문서 수: {len(actual)}",
        f"- 비교 키워드 수: {len(keywords)}",
        "",
        "## 핵심 지표",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.4f}")
    if pair_rows:
        lines.extend(
            [
                "",
                "## 실제 주민 의견-AI 의견 매칭",
                "",
                "| 실제 주민 의견 | AI 의견 | Cosine Similarity |",
                "|---|---|---:|",
            ]
        )
        for row in pair_rows[:15]:
            lines.append(
                f"| {row['actual_opinion']} | {row['ai_opinion']} | {float(row['cosine_similarity']):.2f} |"
            )
    if coverage_rows:
        lines.extend(
            [
                "",
                "## 실제 쟁점별 언급 커버리지",
                "",
                "| 실제 쟁점 | 가장 가까운 AI/토론 쟁점 | Coverage | 매칭된 핵심어 | 누락 핵심어 |",
                "|---|---|---:|---|---|",
            ]
        )
        for row in coverage_rows[:15]:
            lines.append(
                "| {actual} | {ai} | {score:.1f}% | {matched} | {missing} |".format(
                    actual=row["actual_opinion"],
                    ai=row["ai_opinion"],
                    score=float(row["coverage_percent"]),
                    matched=row["matched_terms"],
                    missing=row["missing_terms"],
                )
            )
    lines.extend(["", "## 가상 응답 주요 키워드", ""])
    for keyword, count in keyword_counter.most_common(15):
        lines.append(f"- {keyword}: {count}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.ai_source in {"generated", "both"} and not args.responses:
        raise SystemExit("--responses is required when --ai-source is generated or both.")
    if args.ai_source in {"debate", "both"} and not args.ai_debate_result:
        raise SystemExit("--ai-debate-result is required when --ai-source is debate or both.")

    responses = Path(args.responses) if args.responses else None
    cluster_dir = Path(args.cluster_dir) if args.cluster_dir else None
    generated: list[TextRecord] = []
    if args.ai_source in {"generated", "both"} and responses:
        generated.extend(load_generated(responses, cluster_dir))
    if args.ai_source in {"debate", "both"} and args.ai_debate_result:
        generated.extend(load_generated_from_debate(Path(args.ai_debate_result), args.debate_scope))

    terms = split_csv_terms(args.terms)
    if args.actual_csv:
        actual = load_actual_from_csv(
            Path(args.actual_csv),
            args.actual_text_col,
            args.actual_id_col,
            args.actual_keyword_col,
            args.actual_match_terms_col,
        )
    elif args.actual_urls_file:
        actual = load_actual_from_urls_file(Path(args.actual_urls_file))
    elif args.reference_report_data:
        actual = load_reference_from_report_data(Path(args.reference_report_data))
    else:
        actual = load_actual_from_sqlite(Path(args.sqlite), terms, args.limit)

    if not generated:
        raise SystemExit("No generated response text found.")
    if not actual:
        raise SystemExit("No actual public-opinion text found.")

    generated_keywords = [kw for kw, _ in Counter(r.keyword for r in generated if r.keyword).most_common(20)]
    keywords = split_csv_terms(args.keywords) or [*terms, *generated_keywords]
    keywords = list(dict.fromkeys(k for k in keywords if k))

    gen_vec = count_keyword_vector(generated, keywords)
    actual_vec = count_keyword_vector(actual, keywords)
    metrics = {
        "keyword_count_cosine": safe_cosine(gen_vec, actual_vec),
        **tfidf_metrics(generated, actual),
    }
    crr_metrics, pair_rows = crr_pair_table(actual, generated, args.crr_threshold)
    coverage_metrics, coverage_rows = term_coverage_table(actual, generated, args.coverage_threshold)
    metrics = {**coverage_metrics, **crr_metrics, **metrics}

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "opinion_similarity_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_keyword_csv(output_dir / "keyword_compare.csv", keywords, gen_vec, actual_vec)
    write_pair_csv(output_dir / "opinion_pair_similarity.csv", pair_rows)
    write_coverage_csv(output_dir / "issue_coverage.csv", coverage_rows)
    write_actual_sources_csv(output_dir / "actual_sources.csv", actual)
    write_report(output_dir / "similarity_report.md", metrics, generated, actual, keywords, pair_rows, coverage_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"generated_count={len(generated)}")
    print(f"actual_count={len(actual)}")
    for key, value in metrics.items():
        print(f"{key}={value:.4f}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
