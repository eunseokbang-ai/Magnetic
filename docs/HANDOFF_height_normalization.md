# 인수인계 — 비행 고도 정규화 (다음 작업)

새 대화창에서 이 작업을 시작할 때 이 문서 하나만 읽으면 된다. 이 문서는 2026-09-29 기준이며,
작업이 끝나면 내용을 `TECHNICAL_MANUAL.md`로 옮기고 이 파일은 지운다.

## 0. 이 프로젝트에서 일하는 방식 (지켜야 할 것)

- 사용자는 한국어로 작업하며, 답변도 한국어로 한다.
- **push는 사용자가 요청할 때만** 한다. 커밋은 기능 단위로, 커밋 메시지에 변경 사유와 측정 근거를 적는다.
- 브랜치: `claude/magnetic-processing-improve`, 원격 `https://github.com/eunseokbang-ai/Magnetic.git`.
- 테스트는 **반드시 `backend/` 안에서** 실행한다(시험 자료 경로가 그 기준). 현재 682개 통과.
  ```bat
  cd backend && python -m pytest tests -q
  ```
  Windows 콘솔 인코딩 문제로 `PYTHONIOENCODING=utf-8`을 앞에 붙이면 안전하다.
- 프런트엔드 빌드: `cd frontend && npm run build`. 백엔드는 `frontend/dist`를 그대로 서비스한다.
- 설치본: `packaging/magnetic.spec`(PyInstaller) → `packaging/installer.iss`(Inno Setup, 비밀번호
  `0428683019`). 빌드 명령은 `README.md`에 있다. 설치본을 새로 만들면 `installer.iss`의 `AppVersion`을 올린다.
- 셸에서 여러 줄 파이썬 패치를 heredoc으로 넣으면 따옴표 때문에 자주 깨진다. **패치 스크립트를 파일로
  써서 실행**하는 편이 안전했다.
- 설계 원칙 네 가지(`README.md` 끝): 항상 측정하고 신뢰될 때만 적용, 못 하는 것은 추측하지 않음,
  기본값은 실측으로 정함, 측선 사이는 측정되지 않았음을 명시. 새 기능도 **합성자료(정답 기준)로
  수치를 재고 그 표를 docstring·매뉴얼에 남긴다.**
- 기술 매뉴얼 `docs/TECHNICAL_MANUAL.md`를 고친 뒤 `python docs/build_manual.py`로 PDF를 다시 만든다.
  한글 글꼴에 없는 기호(ᵀ, ‖, ≈ 등)는 `build_manual.py`의 `GLYPH_SUBSTITUTIONS`에 대체 문자를 넣는다.

## 1. 왜 필요한가

프로그램은 측점을 하나의 수평면 위에 있는 것으로 취급한다. 실제 드론은 지형을 따라(약 50 m AGL)
날므로 측점 고도가 수십 m씩 다르고, 그 결과:

- 얕은 이상의 진폭이 고도 차이만큼 달라진다(쌍극자는 거리의 세제곱 — 고도 10 m 차이면 30% 이상).
- 파생 격자(FFT), 오일러, 심도추정, 3D 역산이 모두 "같은 높이에서 잰 값"을 전제한다. 지금은
  `euler_deconvolution.py`와 `inversion.py`가 평면 근사임을 docstring에 밝혀 두었을 뿐이다.
- 고도 차이가 측선과 함께 변하면 측선 줄무늬로도 나타난다.

목표: 측점의 실제 3차원 위치에 등가 소스층을 맞춰, **일정 고도면에서의 값**을 계산해 측점 값으로
대체한다(측점의 x, y는 유지 → 이후 파이프라인은 손대지 않아도 된다).

## 2. 이미 있는 것 (재사용 대상)

| 있음 | 위치 | 비고 |
|---|---|---|
| 측점 고도 | `processed["altitude_ellipsoidal_m"]` | 로더가 MSL + 지오이드로 만든다. 비어 있을 수 있음(NaN) → 그때는 기능 비활성 |
| DEM 적재 | `store.load_dem`, `processing/terrain.py::load_dem_geotiff` | 있으면 AGL을 계산해 보고할 수 있다(정규화에 필수는 아님) |
| 고정 AGL 가정 | `terrain.py::estimate_ground_elevation` | 3D 역산이 쓰는 근사 |
| 쌍극자 커널 (관측면 z=0) | `processing/source_removal.py::_kernel(x, y, sources, f)` | 관측점 z를 받도록 확장하면 된다. `rz = z_obs − z_src` |
| 자유/유도 쌍극자 최소제곱, 감쇠 | `source_removal.py::_solve`, `magnetization.py::_fit` | 상대 정규화(λ = 최대 특이값의 1%) 방식 |
| 격자 기반 등가층 (FFT, CG) | `processing/continuation.py::_fit_source_layer`, `equivalent_source_field` | 관측면이 하나인 격자용. 3D 점에는 못 쓴다 |
| 처리 사슬 훅 | `store.py::_flight_levelled_base` → `_base_without_structures` → `_rebuild_processed` | 정규화는 **`_flight_levelled_base` 다음, 구조물 차감 앞**에 넣는다(레벨링된 자료를 맞추고, 구조물 모델은 정규화된 값에 맞춘다) |
| 되돌리기·저장 패턴 | `repeat_pass_offsets`(상태), `apply_repeat_pass_leveling`(apply/reset), 번들 meta에 저장, `run_pipeline`에서 초기화 | 같은 패턴으로 `height_normalization` 상태를 둔다 |
| 패널 패턴 | `frontend/src/components/RepeatPassPanel.jsx` | 분석 → 적용/되돌리기 → 요약 표시 |

## 3. 설계 제안

**모델**: 유도 방향(IGRF 복각·편각)의 쌍극자를 규칙 격자로 놓은 등가층. 격자 간격 = 측선 간격의
1/2, 깊이 = 가장 낮은 측점 고도 − (측선 간격 × 1.0~1.5). 미지수는 쌍극자 세기(스칼라).
잔류자화는 무시한다 — 정규화 목적에서는 관측을 재현하는 것이 전부이고, 유도 방향 고정이 해를 안정시킨다.

**관측**: 측점을 셀(격자 셀 크기, 기본 10 m)로 블록평균해 수만 개로 줄인다(96.5만 → 약 3~5만).
각 블록의 z는 그 안 측점 고도의 평균.

**풀이**: 정규방정식 `(GᵀG + λ²I) m = Gᵀd`를 **켤레기울기(CG)** 로 푼다. G는 (관측 수 × 소스 수)라
밀집 행렬로 두면 수 GB가 되므로, `G v`와 `Gᵀ v`를 **청크 단위로 계산**하는 matrix-free 방식으로 한다
(numba가 이미 의존성에 있다; `choclo`에도 쌍극자 커널이 있다). 30 반복 이내로 수렴하는지 합성자료로 확인.

**예측**: 각 측점(원래 x, y)에 대해 z = z_ref(기본: 측점 고도의 중앙값; 사용자 지정 가능)에서의 값을
계산해 `anomaly`와 `tmi`를 대체한다. 원래 값과의 차이(정규화 보정량)를 함께 보관해 화면에 통계로 보여준다.

**보고할 것**: 측점 고도 범위·표준편차, z_ref, 등가층 깊이·소스 수, 맞춤 오차(nT rms)와 자료 변동 대비
비율, 보정량 통계(중앙값·최대), 수렴 반복 수. 맞춤 오차가 자료 변동의 30%를 넘으면 경고.

**되돌리기**: `repeat_pass_offsets`와 같은 방식 — 상태에 정규화 결과(측점별 보정량 배열 또는 재계산
파라미터)를 두고 `_rebuild_processed`가 적용, reset으로 제거, 번들 meta에 저장, `run_pipeline`에서 초기화.

## 4. 검증 계획 (테스트로 남길 것)

1. **합성**: 깊이 40 m 쌍극자 + 광역 지질(1.4 km 파장) 자료를 언덕 지형 위 50 m AGL(고도 변화 ±30 m)에서
   50 m 간격 측선으로 샘플링. 정규화 후 값을 **참 모델을 z_ref에서 계산한 값**과 비교 → rms 오차가
   정규화 전보다 크게(예: 1/3 이하로) 줄어야 한다.
2. **평탄 비행에서는 거의 바꾸지 않는다**: 고도 변화 ±1 m면 보정량 중앙값이 잡음 수준.
3. **되돌리기 정확성**: apply → reset 후 원래 값과 동일.
4. **저장·복구**: 번들 왕복 후 정규화 상태 유지.
5. **성능**: 해남 74개 파일(96.5만 점)에서 수 분 이내. `HaeNam_Mag/Magnetometer/*.csv`가 저장소에 있다
   (커밋 대상 아님).
6. **구조물 차감과의 순서**: 정규화 적용 뒤 구조물 모델 차감·되돌리기가 각각 독립적으로 동작.

## 5. 화면

"5-1. 비행 고도 정규화" 패널(7번 수동 편집 위쪽): 고도 통계 표시 → z_ref 입력(기본 중앙값) →
"정규화 적용" / "되돌리기" → 결과 요약과 경고. 적용 여부를 `process_summary`에 넣고 보고서 3장(적용된
보정)에 한 줄 추가. QC 인증서는 건드리지 않는다.

## 6. 참고 문헌

- Dampney (1969) equivalent source technique; Cordell (1992) "A scattered equivalent-source method for
  interpolation and gridding of potential-field data in three dimensions" — 3차원 산재 관측점의 등가층이
  바로 이 문제다.
- Li & Oldenburg (2010) rapid construction of equivalent sources; Soler & Uieda (2021) gradient-boosted
  equivalent sources(대용량 관측에서 블록별 순차 적합) — 관측 수만 개가 넘으면 이 방식을 참고.
