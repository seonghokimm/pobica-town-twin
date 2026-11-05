import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping


UTF8_SIG = "utf-8-sig"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_PATH = SCRIPT_DIR / "hwasun_persona_policy_responses.json"
DEFAULT_TARGET_CSV_PATH = SCRIPT_DIR / "satisfaction" / "childcare_5.csv"

SCORE_COLUMNS = {
    1: "score_1",
    2: "score_2",
    3: "score_3",
    4: "score_4",
    5: "score_5",
}
AGE_GROUP_ORDER = ["15~29세", "30~39세", "40~49세", "50~59세", "60~69세", "70세이상"]
REGION_GROUP_ORDER = ["읍권", "동부권", "서부권"]
REGION_GROUPS = {
    "읍권": {"화순읍"},
    "동부권": {"이서면", "백아면", "동복면", "사평면", "동면"},
    "서부권": {"한천면", "춘양면", "청풍면", "이양면", "능주면", "도곡면", "도암면"},
}
REGION_ALIASES = {
    "청양면": "춘양면",
}


def probability_vector(values: Mapping[str | int, float]) -> list[float]:
    total = sum(_get_number(values, score) for score in range(1, 6))
    if total <= 0:
        return [0.0] * 5
    return [_get_number(values, score) / total for score in range(1, 6)]


def wasserstein_distance(
    score_distribution: Mapping[str | int, float],
    target_ratios: Mapping[str | int, float],
) -> float:
    generated_probs = probability_vector(score_distribution)
    target_probs = probability_vector(target_ratios)
    if not any(generated_probs) or not any(target_probs):
        return 0.0

    generated_cdf = 0.0
    target_cdf = 0.0
    distance = 0.0
    for idx in range(4):
        generated_cdf += generated_probs[idx]
        target_cdf += target_probs[idx]
        distance += abs(generated_cdf - target_cdf)
    return distance


def _get_number(values: Mapping[str | int, float], score: int) -> float:
    return float(values.get(str(score), values.get(score, 0.0)))


def random_score_distribution(total: int, rng: random.Random) -> dict[str, int]:
    counts = {str(score): 0 for score in range(1, 6)}
    for _ in range(total):
        counts[str(rng.randint(1, 5))] += 1
    return counts


def estimate_random_baseline(
    total: int,
    target_ratios: Mapping[str | int, float],
    trials: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    distances = [
        wasserstein_distance(random_score_distribution(total, rng), target_ratios)
        for _ in range(trials)
    ]
    distances.sort()
    mean = sum(distances) / len(distances)
    p05 = distances[int(len(distances) * 0.05)]
    p95 = distances[min(len(distances) - 1, int(len(distances) * 0.95))]
    return mean, p05, p95


def load_result(path: str | Path) -> dict:
    with Path(path).open("r", encoding=UTF8_SIG) as f:
        return json.load(f)


def load_target_table(csv_path: str | Path) -> dict[str, dict[str, float]]:
    with Path(csv_path).open("r", encoding=UTF8_SIG, newline="") as f:
        rows = list(csv.reader(f))

    header_idx = next(
        (
            idx
            for idx, row in enumerate(rows)
            if row and row[0].strip() == "구분" and all(column_name in row for column_name in SCORE_COLUMNS.values())
        ),
        None,
    )
    if header_idx is None:
        raise SystemExit("CSV에서 '구분/score_1/score_2/score_3/score_4/score_5' 헤더를 찾지 못했습니다.")

    header = [cell.strip() for cell in rows[header_idx]]
    column_indexes = {}
    for score, column_name in SCORE_COLUMNS.items():
        try:
            column_indexes[score] = header.index(column_name)
        except ValueError as exc:
            raise SystemExit(f"CSV에 '{column_name}' 컬럼이 없습니다.") from exc

    targets: dict[str, dict[str, float]] = {}
    for row in rows[header_idx + 1:]:
        if not row or not row[0].strip():
            continue
        label = row[0].strip()
        targets[label] = {
            str(score): _parse_float(row[column_idx] if column_idx < len(row) else "")
            for score, column_idx in column_indexes.items()
        }
    return targets


def _parse_float(value: str) -> float:
    value = str(value).strip().replace(",", "")
    if not value:
        return 0.0
    return float(value)


def age_group(age: Any) -> str | None:
    try:
        parsed_age = int(age)
    except (TypeError, ValueError):
        return None
    if 15 <= parsed_age <= 29:
        return "15~29세"
    if 30 <= parsed_age <= 39:
        return "30~39세"
    if 40 <= parsed_age <= 49:
        return "40~49세"
    if 50 <= parsed_age <= 59:
        return "50~59세"
    if 60 <= parsed_age <= 69:
        return "60~69세"
    if parsed_age >= 70:
        return "70세이상"
    return None


def region_group(region_name: Any) -> str | None:
    region = str(region_name or "").strip()
    region = REGION_ALIASES.get(region, region)
    for group_name, regions in REGION_GROUPS.items():
        if region in regions:
            return group_name
    return None


def score_counts() -> dict[str, int]:
    return {str(score): 0 for score in range(1, 6)}


def group_generated_results(payload: dict, group_fn, ordered_labels: list[str]) -> dict[str, dict[str, Any]]:
    grouped = {
        label: {"persona_count": 0, "score_distribution": score_counts()}
        for label in ordered_labels
    }
    for result in payload.get("results", []):
        label = group_fn(result)
        if label not in grouped:
            continue
        score = str(result.get("satisfaction_score"))
        if score not in grouped[label]["score_distribution"]:
            continue
        grouped[label]["persona_count"] += 1
        grouped[label]["score_distribution"][score] += 1
    return grouped


def build_group_report(
    label: str,
    generated_info: dict[str, Any],
    target_ratios: Mapping[str | int, float],
    trials: int,
    seed: int,
    threshold: float,
) -> dict:
    distribution = generated_info.get("score_distribution", score_counts())
    persona_count = int(generated_info.get("persona_count", 0))
    target_total = sum(_get_number(target_ratios, score) for score in range(1, 6))
    report = {
        "label": label,
        "persona_count": persona_count,
        "distribution": distribution,
        "target_ratios": dict(target_ratios),
        "threshold": threshold,
        "skipped": False,
        "skip_reason": "",
    }
    if persona_count <= 0:
        report["skipped"] = True
        report["skip_reason"] = "생성 결과에 해당 집단 페르소나가 없습니다."
        return report
    if target_total <= 0:
        report["skipped"] = True
        report["skip_reason"] = "CSV의 해당 집단 기준 분포가 모두 0이라 검증 기준으로 사용할 수 없습니다."
        return report

    wsim = wasserstein_distance(distribution, target_ratios)
    wrandom, wrandom_p05, wrandom_p95 = estimate_random_baseline(
        total=persona_count,
        target_ratios=target_ratios,
        trials=trials,
        seed=seed,
    )
    wratio = wsim / wrandom if wrandom else float("inf")
    report["wasserstein"] = {
        "sim": wsim,
        "random_mean": wrandom,
        "random_p05": wrandom_p05,
        "random_p95": wrandom_p95,
        "ratio": wratio,
        "improvement_pct": (1 - wratio) * 100,
        "passed": wsim <= wrandom_p05,
        "pass_criterion": "Wsim <= Wrandom_p05",
    }
    return report


def build_csv_wasserstein_reports(
    result_path: str | Path = DEFAULT_RESULT_PATH,
    target_csv: str | Path = DEFAULT_TARGET_CSV_PATH,
    trials: int = 10000,
    seed: int = 42,
    threshold: float = 0.3,
) -> dict[str, Any]:
    payload = load_result(result_path)
    targets = load_target_table(target_csv)
    summary = payload.get("summary", {})
    overall_distribution = summary.get("score_distribution", {})
    overall_count = int(
        summary.get("persona_count")
        or sum(_get_number(overall_distribution, score) for score in range(1, 6))
    )

    age_groups = group_generated_results(
        payload,
        lambda result: age_group(result.get("age")),
        AGE_GROUP_ORDER,
    )
    region_groups = group_generated_results(
        payload,
        lambda result: region_group(result.get("region_name_kr")),
        REGION_GROUP_ORDER,
    )

    return {
        "result_path": str(Path(result_path).resolve()),
        "target_csv": str(Path(target_csv).resolve()),
        "threshold": threshold,
        "overall": build_group_report(
            "전체(2025년)",
            {"persona_count": overall_count, "score_distribution": overall_distribution},
            targets.get("2025년", {}),
            trials,
            seed,
            threshold,
        ),
        "age_groups": [
            build_group_report(label, age_groups[label], targets.get(label, {}), trials, seed, threshold)
            for label in AGE_GROUP_ORDER
        ],
        "region_groups": [
            build_group_report(label, region_groups[label], targets.get(label, {}), trials, seed, threshold)
            for label in REGION_GROUP_ORDER
        ],
    }


def print_csv_wasserstein_reports(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("CSV 기준 Wasserstein 검증")
    print("=" * 72)
    print(f"결과 파일: {report['result_path']}")
    print(f"기준 CSV: {report['target_csv']}")
    print("판정 기준: Wsim <= Wrandom 5% 지점")
    print("-" * 72)
    _print_group_report(report["overall"])
    print("\n[연령대별]")
    for group_report in report["age_groups"]:
        _print_group_report(group_report)
    print("\n[권역별]")
    print("읍권: 화순읍")
    print("동부권: 이서면, 백아면, 동복면, 사평면, 동면")
    print("서부권: 한천면, 춘양면, 청풍면, 이양면, 능주면, 도곡면, 도암면")
    for group_report in report["region_groups"]:
        _print_group_report(group_report)
    print("=" * 72)


def _print_group_report(report: dict[str, Any]) -> None:
    print(f"[{report['label']}] 표본 수: {report['persona_count']}명")
    print(f"  생성 분포(명): {report['distribution']}")
    print(f"  실제 분포(%): {report['target_ratios']}")
    if report["skipped"]:
        print(f"  판정: SKIP - {report['skip_reason']}")
        return
    w = report["wasserstein"]
    sample_note = " / 참고용(n<20)" if report["persona_count"] < 20 else ""
    print(
        f"  Wasserstein={w['sim']:.4f}, "
        f"Wrandom={w['random_mean']:.4f} (5~95%: {w['random_p05']:.4f}~{w['random_p95']:.4f}), "
        f"Wsim/Wrandom={w['ratio']:.4f}, "
        f"개선율={w['improvement_pct']:.1f}%, "
        f"기준={w['pass_criterion']}, "
        f"{'PASS' if w['passed'] else 'FAIL'}{sample_note}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare persona satisfaction distributions against CSV targets using Wasserstein distance."
    )
    parser.add_argument("--result", default=str(DEFAULT_RESULT_PATH), help="결과 JSON 경로")
    parser.add_argument("--target-csv", default=str(DEFAULT_TARGET_CSV_PATH), help="실제 만족도 CSV 경로")
    parser.add_argument("--trials", type=int, default=10000, help="랜덤 baseline 반복 횟수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="이전 버전 호환용 인자입니다. 현재 PASS 기준은 Wsim <= Wrandom_p05입니다.",
    )
    args = parser.parse_args()
    report = build_csv_wasserstein_reports(
        result_path=args.result,
        target_csv=args.target_csv,
        trials=args.trials,
        seed=args.seed,
        threshold=args.threshold,
    )
    print_csv_wasserstein_reports(report)


if __name__ == "__main__":
    main()
