# MineMag — 드론 자력탐사 지뢰·불발탄 탐지

SENSYS MagDrone R3(2탐침 플럭스게이트 어레이) 자료로 지뢰·불발탄 같은 근지표 소형 표적을 찾는
전용 프로그램이다. 처리 엔진은 저장소 최상위의 공용 라이브러리 `magcore/`를 DroneMag Studio와
함께 쓰고, 파이프라인·화면·설치본은 따로 둔다 — 자료 모델(탐침 2개 × 3축 벡터 × 트랙)과 처리
단계(플럭스게이트 보정, 어레이 레벨링, 구배, 0.1 m 격자, 표적 피킹, 소탕 목록)가 광물탐사와
절반 이상 다르기 때문이다. 왜 분리했는지와 시험 자료 검토는 `C:\magnetic\Landmine\검토_*.md`.

## 구성

```
minemag/
  backend/
    app/
      io_/magdrone_r3.py   R3 ASC 적재 — 두 탐침·트랙·3축·고도를 모두 보존, 파일명의 날짜와
                           ms-since-midnight를 합쳐 실제 시각 생성, 적재 요약(탐침 간격·샘플링·
                           공백·속도·잡음·탐침 오프셋)
      store.py             프로젝트(메모리), UTM 투영은 magcore.lines 재사용
      main.py              FastAPI: 프로젝트 생성 → 업로드 → 요약 → 지도용 점
      paths.py             데이터 폴더(체크아웃: backend/.data, 설치본: %LOCALAPPDATA%\MineMag)
    tests/                 실제 시험장 자료 두 트랙을 고정 자료로 씀
    launcher.py            포트 8100부터 빈 포트를 골라 띄우고 브라우저를 연다
  frontend/                업로드 → 요약 → 지도. DroneMag Studio와 같은 Vite + React + Leaflet
```

## 실행

저장소 루트의 `run_minemag.bat`을 더블클릭한다. 가상환경(`backend\venv`, DroneMag Studio와 공유)과
프런트 빌드가 없거나 커밋이 바뀌었으면 설치·빌드를 먼저 하고, 8100번부터 빈 포트를 골라 서버를 띄운 뒤
브라우저를 연다. 코드를 고치고 커밋 없이 바로 보려면 `.last_build_commit_minemag`을 지우고 다시 실행한다.

개발 중 수동 실행:

```bat
cd minemag\backend && python -m pytest tests -q
cd minemag\frontend && npm install && npm run build
cd minemag\backend && python launcher.py
```

개발 중에는 `npm run dev`(5174)가 `/api`를 8100으로 넘긴다.

## 있는 기능

1. 적재 요약(`io_/magdrone_r3.py`)
2. 플랫폼 잡음 QC(`processing/platform_qc.py`): 조용한 구간의 배경 잡음(전대역·>1 Hz·>5 Hz), 두 탐침 잡음 상관과
   구배 상쇄율, 잡음 스펙트럼과 봉우리의 공간 주기, 판정문. 조용한 구간의 문턱은 자료 자체의 강건 표준편차에
   맞춰 올라가므로 보행 잡음이 큰 플랫폼에서도 측정이 된다
3. 격자(`processing/gridding.py`): `magcore.gridding` 선형 보간, 셀 0.1 m, 가장 가까운 측점에서 0.6 m 넘는 셀은 비움,
   탐침·파일별 중앙값 제거. 지도에 세 모서리로 놓는 회전 오버레이
4. 반복 측정 비교: 파일 쌍마다 같은 셀의 블록 평균을 비교(보간 없음 — 근거는 모듈 docstring의 표). 차의 rms가 반복도
5. 표적 피킹(`processing/targets.py`): 측정(파일)마다 따로 격자의 국소 봉우리를 찾아 `magcore.dipole_fit`의 유도 쌍극자를
   원 측점에 맞추고(위치·탐침 아래 깊이·모멘트), 문턱은 그 측정의 조용한 구간 잡음 rms × k(기본 5, 최소 20 nT).
   측정이 여럿이면 0.5 m 안에서 맞춰 "n회 중 k회에서 보임"을 세고, 등급 A/B/C(모든 측정에서 보임·SNR·맞춤)를 매긴다.
   파일별 탐침 지상고로 매설 심도를 낸다. 복각·편각은 IGRF. CSV 내보내기(`/targets.csv`, 위경도·UTM·등급·심도·크기).
   시험장 실자료: 사각형 안 표적 4개와 남쪽 1개가 드론·로봇 2회 모두에서 잡혀 A, 드론에서만 보이는 약한 것 35개는 B

## 다음 단계

1. 소탕 목록의 현장 형식(GPS 웨이포인트·KML)과 보고서
2. 플럭스게이트 보정(오프셋·스케일·비직교성)과 어레이 레벨링, 수평 구배 — 회전 캘리브레이션 자료가 있어야 검증 가능
3. 등급 규칙의 재검토: 드론만 보는 약한 표적이 로봇 측정 때문에 A가 못 되는 경우(플랫폼이 다르면 "보임"의 분모를 플랫폼별로)
