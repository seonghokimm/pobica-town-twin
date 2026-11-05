**Town Twin 페르소나 구축 — 실행 가이드**

Nemotron-Personas-Korea 기반 화순군 synthetic citizen 100인 구축 (보면서 바로 따라 하는 버전)

이 문서는 페르소나 구축을 코드와 함께 단계별로 따라 할 수 있게 정리한 실행 가이드다. 팀원(배*정) 설계 문서의 확정 사항(19세 이상 성인 모집단, 실측 컬럼, 외부인 RAG 처리)을 반영했다.

**0\. 시작 전 — 반드시 기억할 3가지**

**① 골격 vs 기준 vs 근거 분리:** NVIDIA persona \= 사람의 기본 프로필(골격) / 화순 인구데이터 \= 몇 명 뽑을지 정하는 비율 기준 / RAG \= 반응 시 참고하는 근거. 셋을 섞어 새 사람을 만들지 않는다.

**② 1,230명은 후보군이지 최종이 아니다:** "화순 persona 1,230명 생성"이 아니라 "화순 거주자로 추정되는 후보군 1,230명 필터링". 최종 100명은 실제 인구비율로 다시 추출.

**③ 재현성:** random\_seed=42 고정. 같은 코드 \= 같은 100명. 발표·보고서에서 재현 가능해야 한다.

**1\. 모집단 정의 (확정)**

최종 모집단 \= "화순군 행정정책의 영향을 직접 받는 화순군 거주 성인 주민(만 19세 이상)".

- HF 화순 후보군의 실제 연령 범위가 19\~98세라, 15\~18세 후보는 존재하지 않는다.

- 인구데이터는 5세 단위(15-19, 20-24 …)로 제공되므로 구간명은 그대로 유지한다.

- 단, 15-19 구간의 실제 선발 후보는 후보군 특성상 19세 persona로 채워진다. (구간명 유지 \+ 후보 한계 명시)

- 미성년자(아동·청소년)는 직접 agent로 세우지 않고, 부모·조부모·보육가구 등 성인 가구 persona로 간접 반영한다.

**보고서 문장:** 실제 화순군 인구데이터가 5세 단위로 제공되므로 동일 연령구간을 유지하였다. 다만 HF 화순 후보군은 19세 이상으로 구성되어, 15-19세 구간의 실제 선발 후보는 19세 persona로 구성된다.

**2\. 데이터셋 실측 정보**

**데이터셋:** nvidia/Nemotron-Personas-Korea (split: train, 100만 행). 라이선스 CC BY 4.0.

팀에서 실제 확인한 주요 컬럼(이 이름 그대로 코드에 사용):

uuid, persona, sex, age, marital\_status, family\_type, housing\_type, education\_level, occupation, district, province, country

화순 후보군 필터: district 또는 province에 "화순" 포함. 행 기준 카운트(양쪽에 다 들어가도 1명). 확인 결과 후보군 \= 1,230명, 값 예시 district="전라남-화순군", province="전라남".

**3\. 전체 구축 흐름 한눈에**

1\.   HF 로드 → 화순 후보군 1,230명 필터

2\.   후보군에 5세 age\_group 부여

3\.   화순 실제 인구데이터로 연령대별 quota 계산 (합 100, largest remainder)

4\.   quota만큼 후보군에서 층화추출 (seed=42)

5\.   부족 구간 인접 보충 \+ 로그

6\.   원본 필드 유지 \+ 정책반응용 보조변수 부여

7\.   RAG 연결 → 정책 시나리오별 반응 생성

8\.   최종 100명 분포 검증 (실제 인구 대비 MAE/χ²)

**4\. 단계별 구현 (코드 포함)**

**STEP 1 — HF 로드 & 화순 후보군 필터**

from datasets import load\_dataset  
import pandas as pd, numpy as np  
   
ds \= load\_dataset('nvidia/Nemotron-Personas-Korea', split='train')  
df \= ds.to\_pandas()  
   
\# 화순 포함 행 필터 (district 또는 province)  
mask \= (df\['district'\].str.contains('화순', na=False) \|  
        df\['province'\].str.contains('화순', na=False))  
cand \= df\[mask\].drop\_duplicates('uuid').reset\_index(drop=True)  
print('화순 후보군:', len(cand))   \# 약 1,230  
**주의:** drop\_duplicates(uuid)로 중복 제거. 결과 수가 1,230과 크게 다르면 필터 조건(컬럼명/값)을 다시 확인.

**STEP 2 — 5세 단위 age\_group 부여**

def to\_age\_group(a):  
    if a \>= 70: return 'age\_70\_over'        \# 70+ 통합(인구데이터 상한과 일치시 사용)  
    lo \= (a // 5\) \* 5  
    return f'age\_{lo}\_{lo+4}'  
   
cand\['age\_group'\] \= cand\['age'\].apply(to\_age\_group)  
**맞춤:** 인구데이터(persona\_population\_base\_5y.csv)의 age\_group 표기와 정확히 일치시킨다. (예: age\_20\_29처럼 10세 묶음이면 그 규칙에 맞춰 함수 수정)

**STEP 3 — 실제 인구데이터로 quota 계산**

보유 파일 persona\_population\_base\_5y.csv 컬럼: region\_code, gender, age\_group, population\_count. 연령대별 합계로 비율을 구한다.

pop \= pd.read\_csv('persona\_population\_base\_5y.csv')  
\# 19세 이상만(성인 모집단)으로 한정하려면 15\_19 처리 정책에 맞게 필터  
age\_pop \= pop.groupby('age\_group')\['population\_count'\].sum()  
ratio \= age\_pop / age\_pop.sum()  
   
\# largest remainder method로 합계 100 보정  
raw \= ratio \* 100  
floor \= np.floor(raw).astype(int)  
rem \= 100 \- floor.sum()  
order \= (raw \- floor).sort\_values(ascending=False).index  
quota \= floor.copy()  
for g in order\[:rem\]:  
    quota\[g\] \+= 1  
assert quota.sum() \== 100  
print(quota)

**STEP 4 — 층화추출 (seed 고정)**

rng \= np.random.default\_rng(42)  
picked \= \[\]  
log \= \[\]  
for g, need in quota.items():  
    pool \= cand\[cand\['age\_group'\] \== g\]  
    if len(pool) \>= need:  
        sel \= pool.sample(n=need, random\_state=42)  
    else:   \# STEP 5 보충 로직으로 처리  
        sel \= pool.copy()  
        log.append((g, need, len(pool)))  
    picked.append(sel)  
chosen \= pd.concat(picked)

**STEP 5 — 부족 구간 인접 보충 \+ 로그**

특정 연령대 후보가 quota보다 적으면 ① 인접 연령대 → ② 같은 대분류(고령 등) 순으로 보충하고, 보충 사실을 반드시 기록한다.

def fill\_adjacent(chosen, cand, quota):  
    need\_total \= 100 \- len(chosen)  
    used \= set(chosen\['uuid'\])  
    if need\_total \<= 0: return chosen  
    extra \= cand\[\~cand\['uuid'\].isin(used)\].sample(  
                frac=1, random\_state=42)  
    add \= extra.head(need\_total).copy()  
    add\['supplement\_reason'\] \= 'adjacent\_age\_group\_fill'  
    return pd.concat(\[chosen, add\])  
   
final \= fill\_adjacent(chosen, cand, quota).reset\_index(drop=True)  
assert len(final) \== 100  
**원칙:** 보충은 인접 연령 우선. generated\_supplement(규칙 생성)는 최후 수단이며 source 표시. 모든 보충은 supplement\_reason에 남긴다.

**STEP 6 — 원본 필드 유지 \+ 보조변수 부여**

NVIDIA 원본 필드는 그대로 두고(나이·성별·직업·서술 등), 정책 반응을 일관되게 만들 보조변수(태그/점수)만 규칙으로 추가한다. 이는 새 사람을 만드는 게 아니라 "이 사람이 어떤 정책에 민감할지" 설명하는 태그다.

def add\_traits(row):  
    t \= {}  
    if row\['age'\] \>= 70:  
        t\['mobility\_constraint'\]='높음'; t\['digital\_accessibility'\]='낮음'  
        t\['public\_service\_dependency'\]='높음'  
    if '농' in str(row\['occupation'\]):  
        t\['local\_attachment'\]='높음'  
    if str(row\['family\_type'\]).startswith('1인'):  
        t\['social\_isolation\_risk'\]='높음'  
    return t  
   
final\['policy\_relevant\_traits'\] \= final.apply(  
        lambda r: add\_traits(r), axis=1)  
**확장:** 여기에 슈워츠 기본가치(보존·변화개방성 등)와 화순 사회지표 태도를 같은 방식으로 얹으면 인격 레이어가 완성된다. (별도 인격 부여 가이드 참조)

**STEP 7 — RAG 연결 (반응 시점)**

RAG는 persona에 미리 합치지 않는다. 정책 시나리오가 들어올 때 관련 문서(민원·정책계획·뉴스·조례·읍면별 공약·버스노선·마을지식)를 검색해 agent 프롬프트에 함께 넣는다.

\# 개념 예시  
prompt \= build\_prompt(  
    persona \= final.iloc\[i\],            \# 사람  
    policy  \= policy\_scenario,          \# 정책  
    context \= rag.search(policy\_query)  \# 근거 문서  
)  
reaction \= llm(prompt)   \# 수용도/우려/갈등/찬반 생성

**STEP 8 — 최종 100명 검증**

- 연령대 분포: 최종 100명 vs 실제 화순 인구 → 절대오차·MAE·χ² 적합도

- 성별 분포: 남/여 비율 차이 (연령대×성별 교차도 가능 범위에서)

- 무결성: 총 100명, 결측 없음, uuid 중복 없음

final\_ratio \= final\['age\_group'\].value\_counts(normalize=True)  
mae \= (final\_ratio \- ratio).abs().mean()  
print('연령분포 MAE:', round(mae,4))  
assert final\['uuid'\].is\_unique and len(final)==100

**5\. 최종 persona 파일 컬럼 (권장)**

| 컬럼 | 의미 |
| ----- | ----- |
| **persona\_id** | 프로젝트 내부 고유 ID |
| **source\_uuid** | HF 원본 uuid |
| **source\_dataset** | 원천 데이터셋명 |
| **age / age\_group\_5y** | 실제 나이 / 5세 연령대 |
| **sex / marital\_status / family\_type** | 성별·혼인·가구유형(원본) |
| **housing\_type / education\_level / occupation** | 주거·교육·직업(원본) |
| **base\_persona\_text** | 원본 persona 서술문 |
| **policy\_relevant\_traits** | 정책 반응용 요약 특성(보조변수) |
| **mobility\_constraint / income\_insecurity / …** | 정책 민감도 태그/점수 |
| **change\_acceptance / local\_attachment** | 정책 수용성 · 지역 애착 |
| **complaint\_expression\_tendency** | 민원·불만 표출 경향 |
| **sampling\_weight / sampling\_group** | 추출 가중치 · 추출 기준 그룹 |
| **supplement\_reason / source\_quality** | 보충 사유 · 원천 품질 |

**6\. 외부 persona 처리 원칙 (확정)**

광주 거주자·외부 관광객·귀향 희망자 등 외부인은 agent로 만들지 않는다. 모집단을 섞으면 최종 수용성 점수가 "누구의 점수인지" 불명확해지기 때문이다.

**원칙:** agent 모집단 \= 화순 주민 100명. 외부 요인은 정책 배경·RAG 근거로만 간접 반영.

- 축제 정책 → 외부 관광객 persona 미생성, 화순 주민이 예산·교통혼잡·소음·상권·생활불편 관점에서 반응

- 청년 유입 정책 → 외부 2030 persona 미생성, 화순 주민이 예산 형평성·공동체 수용성 관점에서 반응

- 교통 정책 → 광주 통근자 persona 미생성, 화순 주민이 광주 접근성·병원/시장 접근성 관점에서 반응

**7\. 읍·면 매핑 — 1차는 하지 않음**

데이터셋 지리 단위는 시도·district까지이며 읍면(화순읍·능주면 등) 정보가 없다.

**권장:** 1차 설계는 화순군 단위로 100명 구성. 읍면 차이는 RAG의 "읍면별 공약·버스노선·마을지식"으로 반영. 읍면별 결과가 핵심인 정책에 한해 외부 읍면 통계로 사후 배정하고 "원본엔 읍면 없음, 외부 통계로 배정"을 명시.

**8\. 구현 시 체크리스트 (흔한 실수)**

- 1,230명을 "최종"이라 부르지 않기 → "후보군"이라 표기

- HF 후보군 분포를 그대로 100명에 적용하지 않기 → 실제 인구비율로 quota 재계산

- 인구데이터·RAG를 persona에 미리 합치지 않기 → 비율 기준 / 반응 시 근거로만

- 미성년자 직접 agent화 금지 → 가구 단위로 간접 반영

- random\_seed=42 고정 → 재현성

- 보충(supplement) 발생 시 반드시 로그 → 검증 가능성

- age\_group 표기를 인구데이터와 100% 일치 → quota 매칭 오류 방지

**9\. 평가 체계와 연결**

- Persona 분포 검증: 최종 100명 연령·성별 분포 vs 실제 화순 인구 (MAE, χ² 적합도)

- 정책 반응 검증: agent 반응 분포 vs 실제 만족도·정책 선호·민원 키워드

- RAG 검색 품질: Recall@K, Precision@K, source\_type 적합성, 수작업 검증

**10\. 한 줄 요약**

Nemotron 화순 후보군 1,230명(골격)에서, 실제 화순 5세 인구비율로 100명을 seed 고정 층화추출하고, 원본 필드는 유지한 채 정책 민감도 보조변수를 얹은 뒤, 반응 시점에 RAG 근거를 함께 넣는다. 외부인은 agent가 아닌 RAG로, 읍면은 1차에선 군 단위로 둔다.

