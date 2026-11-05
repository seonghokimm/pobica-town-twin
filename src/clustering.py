"""
화순군 시민 의견 군집화: UMAP + HDBSCAN, DBCV 최대화 기준.

clustering_core.py 입력 파싱/피처 생성 로직을 재사용하되,
하이퍼파라미터 선택 기준을 프로젝트 임의 가중합이 아니라 DBCV 최댓값으로 둡니다.

기본 실행:
    python clustering.py

"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import clustering_core as base


DEFAULT_INPUT_CANDIDATES = (
    "hwasun_persona_policy_responses_d3.json",
    "hwasun_persona_policy_responses_d3.csv",
    "hwasun_persona_policy_responses.json",
    "hwasun_persona_policy_responses.csv",
)


def discover_input(script_dir: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = script_dir / path
        if not path.exists():
            raise SystemExit(f"입력 파일이 없습니다: {path}")
        return path.resolve()

    for name in DEFAULT_INPUT_CANDIDATES:
        path = script_dir / name
        if path.exists():
            return path.resolve()

    fallback = base.discover_default_input(script_dir)
    if fallback is None:
        raise SystemExit("입력 파일을 찾지 못했습니다. --input으로 JSON/CSV/XLSX 파일을 지정해 주세요.")
    return fallback.resolve()


def parse_grid(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("비어 있는 grid 값입니다.")
    return values


def cluster_labels(labels: Any) -> list[int]:
    return sorted({int(label) for label in labels if int(label) != -1})


def label_sizes(labels: Any) -> dict[int, int]:
    return dict(Counter(int(label) for label in labels if int(label) != -1))


def normalized_entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy / math.log(len(counts), 2)


def allocation_entropy(labels: Any) -> float:
    sizes = label_sizes(labels)
    return normalized_entropy_from_counts(list(sizes.values()))


def conditional_entropy(records: list[base.CitizenResponse], labels: Any, field: str) -> float | None:
    pairs: list[tuple[int, str]] = []
    for record, label in zip(records, labels, strict=True):
        if int(label) == -1:
            continue
        value = getattr(record, field, None)
        if value is None and field in record.extra:
            value = record.extra[field]
        value = str(value or "").strip()
        if value:
            pairs.append((int(label), value))

    if not pairs:
        return None

    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    label_counts: Counter[int] = Counter()
    for label, value in pairs:
        grouped[label][value] += 1
        label_counts[label] += 1

    total = sum(label_counts.values())
    weighted_entropy = 0.0
    for label, counter in grouped.items():
        subtotal = label_counts[label]
        entropy = 0.0
        for count in counter.values():
            p = count / subtotal
            entropy -= p * math.log(p, 2)
        weighted_entropy += (subtotal / total) * entropy
    return weighted_entropy


def run_hdbscan_with_probability(
    features: Any,
    *,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[Any, Any]:
    np = base.import_numpy()
    try:
        import hdbscan
    except ModuleNotFoundError as exc:
        raise base.OptionalDependencyError(
            "DBCV 계산과 membership probability 저장을 위해 hdbscan 패키지가 필요합니다.\n"
            "설치 명령: python -m pip install hdbscan"
        ) from exc

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = model.fit_predict(features)
    probabilities = getattr(model, "probabilities_", None)
    if probabilities is None:
        probabilities = np.zeros(len(labels), dtype=float)
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def evaluate_candidate(records: list[base.CitizenResponse], reduced_features: Any, labels: Any, probabilities: Any) -> dict[str, Any]:
    np = base.import_numpy()
    from hdbscan.validity import validity_index
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    labels = np.asarray(labels, dtype=int)
    non_noise = labels != -1
    n_clusters = len(cluster_labels(labels))
    sizes = label_sizes(labels)
    metrics: dict[str, Any] = {
        "cluster_count": n_clusters,
        "noise_ratio": float(np.mean(labels == -1)),
        "min_cluster_member_count": min(sizes.values()) if sizes else None,
        "max_cluster_member_count": max(sizes.values()) if sizes else None,
        "allocation_entropy": allocation_entropy(labels),
        "mean_membership_probability": float(probabilities[non_noise].mean()) if np.any(non_noise) else 0.0,
        "conditional_entropy_keyword": conditional_entropy(records, labels, "summary_keyword"),
        "conditional_entropy_stance": conditional_entropy(records, labels, "stance"),
        "dbcv": None,
        "silhouette": None,
        "davies_bouldin": None,
        "valid": False,
    }

    if n_clusters >= 2 and int(non_noise.sum()) > n_clusters:
        filtered_features = reduced_features[non_noise]
        filtered_labels = labels[non_noise]
        try:
            metrics["dbcv"] = float(
                validity_index(
                    reduced_features.astype("float64"),
                    labels.astype("int64"),
                    metric="euclidean",
                )
            )
            metrics["silhouette"] = float(silhouette_score(filtered_features, filtered_labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(filtered_features, filtered_labels))
            metrics["valid"] = math.isfinite(float(metrics["dbcv"]))
        except Exception as exc:  # noqa: BLE001
            metrics["dbcv_error"] = str(exc)
    return metrics


def cosine_distance_matrix(features: Any) -> Any:
    np = base.import_numpy()
    matrix = np.asarray(features, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-12, None)
    similarity = normalized @ normalized.T
    return 1.0 - similarity


def top_counter(items: list[Any], limit: int = 5) -> list[tuple[str, int]]:
    counter = Counter(str(item or "미분류") for item in items)
    return counter.most_common(limit)


def build_medoid_outputs(
    records: list[base.CitizenResponse],
    original_features: Any,
    reduced_features: Any,
    labels: Any,
    probabilities: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    np = base.import_numpy()
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    total = len(records)
    sizes = label_sizes(labels)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    medoid_rank: dict[int, int] = {}
    medoid_distance: dict[int, float] = {}
    medoid_indexes: set[int] = set()

    for label in cluster_labels(labels):
        indexes = np.flatnonzero(labels == label)
        sub_features = original_features[indexes]
        distances = cosine_distance_matrix(sub_features)
        mean_distances = distances.mean(axis=1)
        order = np.argsort(mean_distances)
        representative_index = int(indexes[int(order[0])])
        medoid_indexes.add(representative_index)

        for rank, local_position in enumerate(order, start=1):
            global_index = int(indexes[int(local_position)])
            medoid_rank[global_index] = rank
            medoid_distance[global_index] = float(mean_distances[int(local_position)])

        group = [records[int(idx)] for idx in indexes]
        reduced_group = reduced_features[indexes]
        centroid = reduced_group.mean(axis=0)
        radius = float(np.max(np.linalg.norm(reduced_group - centroid, axis=1))) if len(indexes) else None
        keyword_counter = Counter(record.summary_keyword or "미분류" for record in group)
        scores = [float(record.satisfaction_score) for record in group]
        representative = records[representative_index]
        summary = {
            "cluster_id": int(label),
            "cluster_name": f"군집 {label}",
            "is_noise": False,
            "size": int(len(indexes)),
            "size_ratio": round(len(indexes) / total, 6),
            "avg_score": round(sum(scores) / len(scores), 6) if scores else None,
            "score_distribution": dict(sorted(Counter(str(int(score)) if score.is_integer() else str(score) for score in scores).items())),
            "top_keywords": top_counter([record.summary_keyword for record in group]),
            "top_regions": top_counter([record.region_name_kr for record in group]),
            "age_groups": top_counter([record.age_group for record in group]),
            "normalized_keyword_entropy": normalized_entropy_from_counts(list(keyword_counter.values())),
            "radius_in_umap_space": round(radius, 6) if radius is not None else None,
            "representative_selection_method": "medoid_in_original_feature_space",
            "representative_persona_id": representative.persona_id,
            "representative_medoid_distance": round(medoid_distance[representative_index], 6),
            "representative_opinion": representative.opinion_text,
        }
        summaries.append(summary)
        representatives.append(
            {
                **asdict(representative),
                "cluster_id": int(label),
                "representative_selection_method": "medoid_in_original_feature_space",
                "representative_medoid_distance": round(medoid_distance[representative_index], 6),
                "membership_probability": float(probabilities[representative_index]),
            }
        )

    for index, record in enumerate(records):
        label = int(labels[index])
        row = asdict(record)
        row.update(
            {
                "cluster_id": label,
                "cluster_name": "noise" if label == -1 else f"군집 {label}",
                "is_noise": label == -1,
                "is_representative": index in medoid_indexes,
                "representative_rank": medoid_rank.get(index),
                "selection_reason": "원본 feature 공간 medoid" if index in medoid_indexes else "",
                "membership_probability": float(probabilities[index]),
                "distance_to_medoid_center": medoid_distance.get(index),
                "cluster_size": sizes.get(label, 0) if label != -1 else int(np.sum(labels == -1)),
                "cluster_size_ratio": round((sizes.get(label, 0) if label != -1 else int(np.sum(labels == -1))) / total, 6),
            }
        )
        rows.append(row)

    return rows, summaries, representatives


def make_output_dir(script_dir: Path, requested: str | None, feature_mode: str) -> Path:
    if requested:
        output_dir = Path(requested).expanduser()
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "outputs" / f"hwasun_clustering_{feature_mode}_dbcv_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding=base.UTF8_SIG)
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding=base.UTF8_SIG, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: base.csv_safe(row.get(key)) for key in fieldnames})


def print_representative_opinions(representatives: list[dict[str, Any]], cluster_summaries: list[dict[str, Any]]) -> None:
    summary_by_cluster = {int(summary["cluster_id"]): summary for summary in cluster_summaries}
    print("\n[대표 에이전트 원문 의견]")
    for representative in sorted(representatives, key=lambda row: int(row["cluster_id"])):
        cluster_id = int(representative["cluster_id"])
        summary = summary_by_cluster.get(cluster_id, {})
        extra = representative.get("extra") or {}
        sex = extra.get("sex") or representative.get("sex") or ""
        occupation = extra.get("occupation") or representative.get("occupation") or ""
        print("=" * 80)
        print(f"군집 {cluster_id} | 크기 {summary.get('size')}명 | 대표 {representative.get('persona_id')}")
        print(
            "지역/나이/성별/직업: "
            f"{representative.get('region_name_kr')} / {representative.get('age')}세 / {sex} / {occupation}"
        )
        print(f"만족도/입장: {representative.get('satisfaction_score')} / {extra.get('stance', '')}")
        print(f"대표 키워드: {representative.get('summary_keyword')}")
        print(f"medoid distance: {representative.get('representative_medoid_distance')}")
        print("의견:")
        print(representative.get("opinion_text"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="화순군 시민 의견 UMAP+HDBSCAN 군집화: DBCV 최대화 기준",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None, help="응답 입력 파일 경로(JSON/JSONL/CSV/XLSX)")
    parser.add_argument("--sheet", type=str, default=None, help="엑셀에서 특정 시트만 분석할 때 지정")
    parser.add_argument("--output-dir", type=str, default=None, help="결과 저장 폴더")
    parser.add_argument(
        "--feature-mode",
        choices=["weighted_blocks", "embedding_only"],
        default="weighted_blocks",
        help="weighted_blocks는 키워드/의견/만족도/페르소나 가중 통합, embedding_only는 API 임베딩만 사용",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="keyword=0.20,opinion=0.65,score=0.05,persona=0.10",
        help="weighted_blocks 모드 가중치",
    )
    parser.add_argument("--openai-embedding-model", type=str, default=base.DEFAULT_OPENAI_EMBEDDING_MODEL)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--embedding-batch-size", type=int, default=100)
    parser.add_argument("--target-cluster-min", type=int, default=3, help="DBCV 최대화 전에 허용할 최소 군집 수")
    parser.add_argument("--target-cluster-max", type=int, default=5, help="DBCV 최대화 전에 허용할 최대 군집 수")
    parser.add_argument("--min-cluster-size-grid", type=str, default="5,8,10,12,14,16,18,20")
    parser.add_argument("--min-samples-grid", type=str, default="1,2,3,5,8")
    parser.add_argument("--umap-neighbors-grid", type=str, default="8,10,12,15,20")
    parser.add_argument("--umap-components-grid", type=str, default="5,8,10")
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main() -> None:
    np = base.import_numpy()
    args = build_arg_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = discover_input(script_dir, args.input)
    output_dir = make_output_dir(script_dir, args.output_dir, args.feature_mode)

    print(f"[1/7] 시민 응답 읽는 중: {input_path}")
    records = base.load_responses(input_path, sheet_name=args.sheet)
    if not records:
        raise SystemExit("파싱된 시민 응답이 없습니다.")
    print(f"      파싱 완료: {len(records)}건")

    print("[2/7] OpenAI 임베딩 준비 중")
    embedder = base.OpenAIEmbeddingClient(
        script_dir=script_dir,
        model_name=args.openai_embedding_model,
        api_key_env=args.api_key_env,
        batch_size=args.embedding_batch_size,
    )

    print(f"[3/7] 피처 생성 중: {args.feature_mode}")
    if args.feature_mode == "weighted_blocks":
        original_features, feature_metadata = base.build_weighted_features(
            records,
            embedder=embedder,
            weights=base.parse_weights(args.weights),
        )
    else:
        original_features, feature_metadata = base.build_embedding_only_features(records, embedder=embedder)

    min_cluster_sizes = parse_grid(args.min_cluster_size_grid)
    min_samples_values = parse_grid(args.min_samples_grid)
    n_neighbors_values = parse_grid(args.umap_neighbors_grid)
    n_components_values = parse_grid(args.umap_components_grid)
    total = len(min_cluster_sizes) * len(min_samples_values) * len(n_neighbors_values) * len(n_components_values)
    print(f"[4/7] DBCV 기준 UMAP + HDBSCAN 그리드 탐색 중: {total}개 조합")

    rows: list[dict[str, Any]] = []
    reduced_cache: dict[tuple[int, int, float], Any] = {}
    checked = 0
    for n_neighbors in n_neighbors_values:
        for n_components in n_components_values:
            cache_key = (int(n_neighbors), int(n_components), float(args.umap_min_dist))
            if cache_key not in reduced_cache:
                reduced_cache[cache_key] = base.reduce_with_umap(
                    original_features,
                    n_components=int(n_components),
                    n_neighbors=int(n_neighbors),
                    min_dist=float(args.umap_min_dist),
                    random_state=int(args.random_state),
                )
            reduced_features = reduced_cache[cache_key]
            for min_cluster_size in min_cluster_sizes:
                for min_samples in min_samples_values:
                    labels, probabilities = run_hdbscan_with_probability(
                        reduced_features,
                        min_cluster_size=int(min_cluster_size),
                        min_samples=int(min_samples),
                    )
                    metrics = evaluate_candidate(records, reduced_features, labels, probabilities)
                    rows.append(
                        {
                            "rank": 0,
                            "selected": False,
                            "feature_mode": args.feature_mode,
                            "umap_n_neighbors": int(n_neighbors),
                            "umap_n_components": int(n_components),
                            "umap_min_dist": float(args.umap_min_dist),
                            "hdbscan_min_cluster_size": int(min_cluster_size),
                            "hdbscan_min_samples": int(min_samples),
                            **metrics,
                        }
                    )
                    checked += 1
                    if checked % args.progress_every == 0 or checked == total:
                        print(f"      탐색 진행: {checked}/{total}")

    valid_rows = [row for row in rows if row["valid"] and row["dbcv"] is not None]
    if not valid_rows:
        raise RuntimeError("DBCV를 계산할 수 있는 유효 군집 후보가 없습니다.")

    constrained_rows = [
        row
        for row in valid_rows
        if int(args.target_cluster_min) <= int(row["cluster_count"]) <= int(args.target_cluster_max)
    ]
    if not constrained_rows:
        available_counts = sorted({int(row["cluster_count"]) for row in valid_rows})
        raise RuntimeError(
            f"군집 수 {args.target_cluster_min}~{args.target_cluster_max}개 조건을 만족하는 후보가 없습니다. "
            f"탐색된 유효 군집 수 후보: {available_counts}"
        )

    constrained_rows.sort(
        key=lambda row: (
            float(row["dbcv"]),
            float(row["silhouette"] if row["silhouette"] is not None else -999.0),
            -float(row["davies_bouldin"] if row["davies_bouldin"] is not None else 999.0),
            -float(row["noise_ratio"]),
        ),
        reverse=True,
    )
    selected = constrained_rows[0]
    selected_key = (
        selected["umap_n_neighbors"],
        selected["umap_n_components"],
        selected["umap_min_dist"],
        selected["hdbscan_min_cluster_size"],
        selected["hdbscan_min_samples"],
    )
    for rank, row in enumerate(constrained_rows, start=1):
        row["rank"] = rank
        key = (
            row["umap_n_neighbors"],
            row["umap_n_components"],
            row["umap_min_dist"],
            row["hdbscan_min_cluster_size"],
            row["hdbscan_min_samples"],
        )
        row["selected"] = key == selected_key

    print("[5/7] 선택 조합으로 최종 HDBSCAN 실행 및 medoid 대표 선정 중")
    selected_reduced = reduced_cache[
        (
            int(selected["umap_n_neighbors"]),
            int(selected["umap_n_components"]),
            float(selected["umap_min_dist"]),
        )
    ]
    selected_labels, selected_probabilities = run_hdbscan_with_probability(
        selected_reduced,
        min_cluster_size=int(selected["hdbscan_min_cluster_size"]),
        min_samples=int(selected["hdbscan_min_samples"]),
    )
    clustered_rows, cluster_summaries, representatives = build_medoid_outputs(
        records,
        original_features,
        selected_reduced,
        selected_labels,
        selected_probabilities,
    )

    print("[6/7] 결과 저장 중")
    write_csv(output_dir / "hyperparameter_search_results.csv", constrained_rows)
    write_csv(output_dir / "clustered_responses.csv", clustered_rows)
    base.write_json(output_dir / "cluster_summary.json", cluster_summaries)
    base.write_json(output_dir / "representative_agents.json", representatives)
    base.write_json(
        output_dir / "run_summary.json",
        {
            "input_file": str(input_path),
            "algorithm": "umap+hdbscan",
            "selection_criterion": (
                f"maximize DBCV among valid candidates with cluster_count "
                f"{args.target_cluster_min}~{args.target_cluster_max}"
            ),
            "feature_mode": args.feature_mode,
            "feature_metadata": feature_metadata,
            "grid": {
                "umap_neighbors_grid": n_neighbors_values,
                "umap_components_grid": n_components_values,
                "umap_min_dist": args.umap_min_dist,
                "hdbscan_min_cluster_size_grid": min_cluster_sizes,
                "hdbscan_min_samples_grid": min_samples_values,
                "searched_combinations": total,
                "valid_candidates": len(valid_rows),
                "cluster_count_constrained_candidates": len(constrained_rows),
                "target_cluster_min": args.target_cluster_min,
                "target_cluster_max": args.target_cluster_max,
                "cached_umap_runs": len(reduced_cache),
            },
            "selected_hyperparameters": {
                "umap_n_neighbors": selected["umap_n_neighbors"],
                "umap_n_components": selected["umap_n_components"],
                "umap_min_dist": selected["umap_min_dist"],
                "umap_metric": "cosine",
                "hdbscan_min_cluster_size": selected["hdbscan_min_cluster_size"],
                "hdbscan_min_samples": selected["hdbscan_min_samples"],
                "hdbscan_metric": "euclidean",
                "hdbscan_cluster_selection_method": "eom",
            },
            "selected_metrics": selected,
            "representative_selection": "medoid in original feature space, excluding noise",
            "outputs": {
                "hyperparameter_search_results": str(output_dir / "hyperparameter_search_results.csv"),
                "clustered_responses": str(output_dir / "clustered_responses.csv"),
                "cluster_summary": str(output_dir / "cluster_summary.json"),
                "representative_agents": str(output_dir / "representative_agents.json"),
            },
        },
    )

    print("[7/7] 완료")
    print("\n[선택 결과: DBCV 최대]")
    print(
        "UMAP(n_neighbors={umap_n_neighbors}, n_components={umap_n_components}, min_dist={umap_min_dist}) / "
        "HDBSCAN(min_cluster_size={hdbscan_min_cluster_size}, min_samples={hdbscan_min_samples})".format(**selected)
    )
    print(f"- DBCV: {selected['dbcv']:.6f}")
    print(f"- clusters: {selected['cluster_count']}")
    print(f"- noise ratio: {selected['noise_ratio']:.2%}")
    print(f"- silhouette: {selected['silhouette']:.6f}")
    print(f"- DBI: {selected['davies_bouldin']:.6f}")
    print(f"- allocation entropy: {selected['allocation_entropy']:.6f}")
    print(f"- 대표 agent 수: {len(representatives)}명")
    print(f"- output: {output_dir}")
    print_representative_opinions(representatives, cluster_summaries)


if __name__ == "__main__":
    try:
        main()
    except base.OptionalDependencyError as exc:
        raise SystemExit(str(exc)) from exc
