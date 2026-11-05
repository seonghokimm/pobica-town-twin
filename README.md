# POBICA (Town Twin)
### 지역주민 페르소나 기반 지자체 정책 사전검증 시뮬레이션

화순군 주민을 LLM 페르소나 100명으로 구성하고, 정책을 시행하기 전에 주민 반응을 시뮬레이션해 보는 프로젝트입니다.
1차 반응을 군집화하고, 대표 페르소나끼리 토론하게 한 뒤, 수정한 정책으로 다시 평가합니다.

> POSCO 청년 AI·Big Data 아카데미 33기 AI 프로젝트 (2026.06~07, 6인 팀)

---

## 역할

| 구분 | 내용 |
|---|---|
| 팀 구성 | 6인 / 김성호 **팀장** |
| **본인 담당** | 화순군 인구데이터 수집·정제, 5세 단위 인구 매핑표 작성 |
| **본인 담당** | NVIDIA Nemotron-Personas-Korea 기반 페르소나 100인 설계, 층화추출, 대표성 검증 ([docs/](docs/)) |
| 팀 공동 참여 | 실측 만족도 대비 시뮬레이션 분포 검증 기준(Wasserstein) 설계·반복 점검 |
| 팀원 담당 | RAG용 SQLite DB 구축, 반응 생성·군집화·토론·재평가 코드 구현, 크롤링·정책 시나리오 |

`src/`의 코드는 팀이 함께 개발한 최종본입니다. 본인이 단독으로 작성한 코드는 아닙니다.

---

## 파이프라인

```
[1] 페르소나 구축     Nemotron 화순 후보 1,230명 → 실제 인구비율로 100명 층화추출   ← 본인 담당
        │
[2] 1차 반응 생성     정책 원문 + 페르소나 + RAG 근거팩(SQLite) → LLM 응답(만족도 1~5, 의견, 키워드)
        │                                                         general_100.py, evidence_pack_builder.py
[3] 분포 검증        시뮬레이션 만족도 분포 vs 실측 사회조사 분포 (Wasserstein, 연령대·권역별)  ← 공동
        │                                                         wasserstein_baseline_check.py
[4] 반응 군집화       임베딩(의견·키워드) + 만족도 + 페르소나 속성 → UMAP + HDBSCAN
        │                                                         clustering.py, clustering_core.py
[5] 대표 토론        군집별 대표(medoid) 에이전트 토론 → 1차 정책 보고서           토론.py
        │
[6] 수정안 재평가     수정 정책으로 100명 재응답 → 대응표본 t-검정 → 2차 보고서       after_100.py
        │
[7] 실제 의견 비교    토론 의견 vs 실제 100원 택시 의견 TF-IDF 코사인 유사도         opinion_similarity_check.py
```

---

## 주요 결과 (저장된 출력 파일 기준)

**페르소나 대표성** ([docs/03](docs/03_persona_validation_report.md))
- 연령대 7개 구간 모두 실제 화순군 인구 대비 오차 1명 미만
- 성비: 실제 남 49.3% / 여 50.7%, 추출 결과 남 49명 / 여 51명
- 결측 0건, uuid 중복 0건

**반응 군집화** ([results/02_clustering/run_summary.json](results/02_clustering/run_summary.json))
- 하이퍼파라미터 600개 조합 탐색 → 군집 3~5개 조건을 만족하는 121개 후보 중 DBCV 최대 조합 선택
- 최종: UMAP `n_neighbors=8, n_components=10` / HDBSCAN `min_cluster_size=5, min_samples=1`
- 군집 3개, 노이즈 0%, DBCV 0.872, silhouette 0.759

**정책 수정 후 재평가** ([results/04_second_report/report_data_2nd.json](results/04_second_report/report_data_2nd.json))
- 대상 정책: 화순군 100원 효도택시 운영 개선안
- 평균 만족도 2.84 → 3.15 (+0.31), 대응표본 t-검정 t(99)=5.04, p≈2×10⁻⁶
- 만족도가 오른 페르소나 23명, 내린 페르소나 0명, 변화 없음 77명

> 이 수치는 모두 LLM 페르소나의 **시뮬레이션 응답**입니다. 실제 주민 반응을 측정한 결과가 아닙니다.

---

## 진행 과정 (주요 의사결정)

| 시기 | 무엇을 바꿨나 | 이유 |
|---|---|---|
| 6월 말 | 화순 인구·설문 데이터 수집, 5세 단위로 재정리 | 페르소나 추출 비율의 기준 데이터 필요 |
| 6/30 | 페르소나를 **임의 생성 → Nemotron 기반 층화추출**로 전환 | 임의 생성 페르소나는 실제 인구구조와 맞는지 검증할 수 없음 ([docs/01](docs/01_nemotron_persona_design.md)) |
| 6/30 | 읍·면 단위 매핑은 1차에서 **보류** | 원천 데이터에 읍면 정보가 없고, 100명을 13개 읍면으로 나누면 읍면당 7~8명이라 통계적 의미가 약함 |
| 6/30 | 외부인(관광객·통근자)은 agent로 만들지 않음 | 모집단을 섞으면 수용성 점수가 누구의 점수인지 불명확해짐 |
| 7/1~2 | 15~18세 제외, 19세 이상 성인 모집단으로 확정 | 원천 데이터가 19세부터 시작 → 인구데이터와 연령 정의를 맞춤 ([docs/03](docs/03_persona_validation_report.md) 4장) |
| 7월 초 | 초기 계획(MAE·χ²)과 MAE 비교 스크립트를 거쳐 검증 지표를 **Wasserstein**으로 통일 | 만족도는 1~5점 순서형 → 평균이 같아도 분포 모양이 다를 수 있음 |
| 7월 초 | 합격 기준을 **랜덤 기준선 대비**로 정의 | Wasserstein은 절대 기준이 없음. 보고서 기준은 `Wsim/Wrandom ≤ 0.3`, 최종 코드는 `Wsim ≤ 무작위 응답 10,000회의 하위 5% 지점` |
| 7월 | 군집화 K-Means 검토 → **UMAP + HDBSCAN** 최종 채택 | 군집 수를 미리 정하지 않고, 비구형·밀도 차이가 있는 의견 분포에 맞추기 위함 |

---

## 기술 스택

- **LLM / 임베딩**: OpenAI API (`gpt-4.1-mini` 기본값, `text-embedding-3-small`)
- **RAG**: SQLite DB에서 SQL로 정책 계획·뉴스·버스노선·마을 지식 조회 → 페르소나별 근거팩 구성
- **분석**: Python, pandas, NumPy, scikit-learn, SciPy (대응표본 t-검정)
- **군집화**: UMAP, HDBSCAN (DBCV, silhouette, Davies-Bouldin, entropy)
- **검증**: Wasserstein distance (+ 랜덤 기준선 몬테카를로), TF-IDF 코사인 유사도

---

## 폴더 구조

```
├── docs/                         페르소나 설계·구축 가이드·검증 보고서 (본인 작성)
├── src/                          파이프라인 코드 (팀 최종본)
├── data/
│   ├── personas/                 최종 페르소나 100명 (Nemotron 기반 합성 데이터)
│   └── policy/                   정책 원문 / 수정안
└── results/
    ├── 01_first_responses/       1차 100명 반응
    ├── 02_clustering/            군집화 결과, 하이퍼파라미터 탐색 결과
    ├── 03_debate_report/         대표 토론 기록, 1차 보고서
    ├── 04_second_report/         수정안 재평가, 2차 보고서
    └── 05_previous_report_reference/
```

---

## 실행 방법

```bash
pip install -r requirements.txt
cp .env.example .env    # OPENAI_API_KEY 입력
```

- 원래는 코드와 데이터를 한 폴더에 두고 실행하던 구조입니다. 이 저장소에서 실행하려면 `--persona-path`, `--db-path`, `--policy-file` 등 경로 인자를 지정해야 합니다.
- RAG용 SQLite DB(`town_twin_full_v3_policy_stage.sqlite3`, 약 67MB)와 실측 만족도 CSV는 저장소에 포함하지 않았습니다(아래 참고). 따라서 이 저장소만으로 전체 파이프라인을 다시 실행할 수는 없습니다.

---

## 저장소에 포함하지 않은 것

| 항목 | 이유 |
|---|---|
| RAG용 SQLite DB | 용량(약 67MB), 크롤링한 뉴스·민원 원문 포함 |
| 실제 100원 택시 의견 CSV | 외부 수집 자료로 재배포 여부가 불분명 |
| 실측 사회조사 만족도 CSV | 원 자료 재배포 여부 확인 전 |
| 페르소나 층화추출 실행 코드 | 원본 파일이 남아 있지 않음. 절차와 코드 예시는 [docs/02](docs/02_persona_build_guide.md)에 정리 |
| `.env` (API 키) | 보안 |

---

## 데이터 출처 및 라이선스

- 페르소나 원천: NVIDIA, *Nemotron-Personas-Korea*, Hugging Face, 2026 — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)
- `data/personas/`의 페르소나는 위 데이터셋에서 추출한 **합성 인물**이며 실존 인물이 아닙니다.

## 한계

- 페르소나는 인구 분포 수준의 근사이며, 실제 화순 주민 개개인의 의견이 아닙니다.
- 정책 수정 후 만족도 상승은 같은 LLM이 수정안을 다시 평가한 결과로, 실제 정책 효과를 의미하지 않습니다.
- 읍·면 단위 차이는 페르소나가 아니라 RAG 근거(읍면별 공약·버스노선 등)로만 반영했습니다.
