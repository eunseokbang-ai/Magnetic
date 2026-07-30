# 드론 자력탐사 자료 처리 프로그램

드론으로 취득한 자력탐사 자료를 업로드해 필터링, 측선 판별, 일변화/IGRF 보정, 그리딩, RTP/RTE/AS/1VD 파생 그리드까지 처리하는 웹 애플리케이션입니다.

## 구성

- `backend/` — FastAPI 처리 서버 (필터, 측선 자동판별, 일변화 보정, IGRF, 그리딩, FFT 변환)
- `frontend/` — React + Leaflet 지도 UI (업로드, 파라미터 조정, 지도 표시, 수동 측선 편집, 그리드/파생그리드 표시)

## 기능

1. 드론 자력 취득 파일 로딩 (Date/Time/Lat/Lon/Mag 등 컬럼 자동 인식, GGA 전용 필드 시간축 보간)
2. 일변화(베이스 자력계) 자료 로딩 (헤더 없는 CSV, 한글 오전/오후 12시간제 지원)
3. 저주파 통과 필터 (Butterworth, 영위상 filtfilt)
4. 이착륙·터닝구간 등 비측선 자료 자동 판별/제거 (측선 방향 허용오차, 터닝 여분 구간(m) 설정 가능)
5. 일변화 보정 + IGRF 보정 (자력이상 = TMI - IGRF), 베이스 로거 시계 오프셋 보정 지원
6. 위치별 자력값(TMI/이상) 컬러 지도 표시
7. GUI 지도에서 측선 체크박스 또는 폴리곤/사각형 드래그로 추가 수동 제거
8. TMI 값 hover 표시 및 통계(min/max/mean/std)
9. 그리딩 (verde 기반 block-mean + 스플라인)
10. RTP / RTE / 1VD / AS 파생 그리드 (Blakely 1995 방식 FFT 필터, 직접 구현)
11. 배경지도 (OSM, Esri 위성, Google 위성) — Leaflet 레이어 컨트롤로 전환

## 실행 방법

### 백엔드

```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

브라우저에서 `http://127.0.0.1:5173` 접속 (Vite 개발서버가 `/api` 요청을 백엔드 8000 포트로 프록시합니다).

### 통합 테스트

실제 샘플 자료(`backend/tests/fixtures/`)로 전체 파이프라인을 API 레벨에서 검증합니다.

```bash
cd backend
python3 tests/test_pipeline_e2e.py
```

## 알려진 제약사항

- **배경지도 타일**: OSM/Esri/Google 타일은 사용자의 브라우저가 직접 요청합니다(백엔드가 프록시하지 않음). 사내망/방화벽 환경에서 해당 타일 서버로의 아웃바운드가 막혀 있으면 지도 배경이 표시되지 않을 수 있습니다.
- **베이스 로거 시계 오프셋**: 베이스 자력계가 GPS 시각에 동기화되지 않은 경우가 흔합니다. 처리 요약에 베이스/드론 시간범위가 겹치지 않는다는 경고가 표시되면 "베이스 시간 오프셋(초)" 값을 조정하세요.
- **Google 위성 타일**은 비공식 XYZ URL이라 이용약관(ToS) 상 상업적 대량 사용에는 적합하지 않을 수 있습니다. 정식 서비스에는 Google Maps Platform API 키 사용을 권장합니다.
- 프로젝트 상태는 서버 메모리에만 저장됩니다(재시작 시 소실). 여러 사용자가 동시에 쓰는 배포 환경에는 별도의 영속 저장소가 필요합니다.
