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

```bat
cd minemag\backend && python -m pytest tests -q
cd minemag\frontend && npm install && npm run build
cd minemag\backend && python launcher.py
```

개발 중에는 `npm run dev`(5174)가 `/api`를 8100으로 넘긴다.

## 다음 단계

1. 플랫폼 잡음 QC: 보행·회전익 잡음 스펙트럼, 두 탐침 상관, 반복 측정 차 — 시험장 자료가 기준 사례
2. 플럭스게이트 보정(오프셋·스케일·비직교성)과 어레이 레벨링, 수평 구배
3. 0.1 m 격자와 덮임(gap) 지도
4. 표적 피킹: 격자 봉우리가 아니라 쌍극자 적합(`magcore.dipole_fit` 재사용), 반복 측정에서 함께 나온 표적의 등급 상향
5. 소탕 목록 내보내기(좌표·심도·크기·등급, GPS 파일), 보고서
