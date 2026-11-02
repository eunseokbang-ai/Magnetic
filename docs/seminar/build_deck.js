// DroneMag Studio seminar deck. Run: node build_deck.js  -> DroneMag_Studio_세미나.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const FIG = path.join(__dirname, "fig");
const fig = (name) => {
  const p = path.join(FIG, name);
  return fs.existsSync(p) ? p : null;
};
const summary = fs.existsSync(path.join(FIG, "andong_summary.json")) ? JSON.parse(fs.readFileSync(path.join(FIG, "andong_summary.json"), "utf8")) : {};
const procSum = fs.existsSync(path.join(FIG, "andong_process_summary.json")) ? JSON.parse(fs.readFileSync(path.join(FIG, "andong_process_summary.json"), "utf8")) : {};
const n = (v, d = 0) => (v === undefined || v === null || Number.isNaN(Number(v)) ? "–" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: d, minimumFractionDigits: d }));

const THEME = {
  name: "DroneMag",
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  colors: {
    dk1: "16213E", lt1: "FFFFFF", dk2: "2C3E50", lt2: "F2F4F7",
    accent1: "C0392B", accent2: "1F4E79", accent3: "E39B2F", accent4: "2E8B57", accent5: "7F8C8D", accent6: "8E44AD",
    hlink: "1F4E79", folHlink: "8E44AD",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "방은석";
pres.title = "DroneMag Studio — 드론 자력탐사 자료처리 프로그램";
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, M = 0.6;

// ------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.2, w: W - 2 * M, h: 1.6, fontSize: 40, bold: true, color: C.background1, align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.9, w: W - 2 * M, h: 1.2, fontSize: 20, color: "CADCFC", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.6, w: W - 2 * M, h: 1.4, fontSize: 38, bold: true, color: C.background1, align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.05, w: W - 2 * M, h: 1.0, fontSize: 18, color: "CADCFC", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: THEME.colors.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: W - 2 * M, h: 0.8, fontSize: 28, bold: true, color: C.text1, align: "left", margin: 0 }, text: "" } },
    { text: { text: "DroneMag Studio 세미나 · KIGAM", options: { x: M, y: H - 0.45, w: 6, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: W - 1.2, y: H - 0.45, w: 0.6, h: 0.3, fontSize: 10, color: THEME.colors.accent5 },
});

// ------------------------------------------------------------ helpers
let secTitle = "";
const section = (t) => { secTitle = t; pres.addSection({ title: t }); };
const content = (title) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: secTitle });
  s.addText(title, { placeholder: "title" });
  return s;
};
const bullets = (s, items, opt = {}) => {
  const x = opt.x ?? M, y = opt.y ?? 1.45, w = opt.w ?? 5.6, h = opt.h ?? 5.2, fs = opt.fontSize ?? 15;
  const runs = [];
  items.forEach((it, i) => {
    if (typeof it === "string") runs.push({ text: it, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } });
    else runs.push({ text: it.text, options: { bullet: it.sub ? { indent: 18 } : true, indentLevel: it.sub ? 1 : 0, bold: !!it.bold, color: it.color, breakLine: i < items.length - 1, paraSpaceAfter: it.sub ? 4 : 8 } });
  });
  s.addText(runs, { x, y, w, h, fontSize: fs, color: C.text2, valign: "top", isTextBox: true, margin: 0, objectName: "본문" });
};
const picture = (s, name, x, y, w, h, caption) => {
  const p = fig(name);
  if (!p) {
    s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.accent5, width: 0.5 }, objectName: "그림 자리" });
    s.addText(`그림 준비 중: ${name}`, { x, y: y + h / 2 - 0.25, w, h: 0.5, fontSize: 12, color: C.accent5, align: "center", isTextBox: true, objectName: "그림 안내" });
  } else {
    s.addImage({ path: p, x, y, w, h, sizing: { type: "contain", w, h }, objectName: name });
  }
  if (caption) s.addText(caption, { x, y: y + h + 0.05, w, h: 0.35, fontSize: 11, color: C.accent5, align: "center", italic: true, isTextBox: true, margin: 0, objectName: "캡션" });
};
const cards = (s, items, opt = {}) => {
  // items: [{head, body, color}], laid out in a row (or grid with cols)
  const cols = opt.cols ?? items.length, x0 = opt.x ?? M, y0 = opt.y ?? 1.5, gap = 0.3;
  const w = opt.w ?? ((W - 2 * M) - gap * (cols - 1)) / cols, h = opt.h ?? 2.1;
  items.forEach((it, i) => {
    const cx = x0 + (i % cols) * (w + gap), cy = y0 + Math.floor(i / cols) * (h + gap);
    s.addShape(pres.ShapeType.roundRect, { x: cx, y: cy, w, h, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: `카드 ${i + 1}` });
    s.addShape(pres.ShapeType.ellipse, { x: cx + 0.2, y: cy + 0.2, w: 0.42, h: 0.42, fill: { color: it.color || C.accent2 }, line: { color: it.color || C.accent2 }, objectName: `카드 아이콘 ${i + 1}` });
    s.addText(String(it.num ?? i + 1), { x: cx + 0.2, y: cy + 0.2, w: 0.42, h: 0.42, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: `카드 번호 ${i + 1}` });
    s.addText(it.head, { x: cx + 0.72, y: cy + 0.18, w: w - 0.9, h: 0.5, fontSize: opt.headSize ?? 15, bold: true, color: C.text1, valign: "middle", isTextBox: true, margin: 0, objectName: `카드 제목 ${i + 1}` });
    s.addText(it.body, { x: cx + 0.2, y: cy + 0.78, w: w - 0.4, h: h - 0.95, fontSize: opt.bodySize ?? 12.5, color: C.text2, valign: "top", isTextBox: true, margin: 0, objectName: `카드 내용 ${i + 1}` });
  });
};
const stat = (s, x, y, w, big, label, color) => {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h: 1.5, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: "통계 카드" });
  s.addText(big, { x, y: y + 0.12, w, h: 0.8, fontSize: 34, bold: true, color: color || C.accent1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: "통계 값" });
  s.addText(label, { x: x + 0.1, y: y + 0.92, w: w - 0.2, h: 0.5, fontSize: 11.5, color: C.text2, align: "center", valign: "top", isTextBox: true, margin: 0, objectName: "통계 설명" });
};
const notes = (s, t) => s.addNotes(t);

// ============================================================ 1. title
section("도입");
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: secTitle });
  s.addText("DroneMag Studio", { placeholder: "title" });
  s.addText("드론 자력탐사 자료처리·해석 프로그램 — 개발 배경, 구성, 안동 적용 사례", { placeholder: "body" });
  s.addText("한국지질자원연구원 · 방은석 · 2026년 11월", { x: M, y: 5.3, w: 8, h: 0.4, fontSize: 14, color: "CADCFC", isTextBox: true, margin: 0, objectName: "발표자" });
  const p = fig("andong_derivatives.png");
  if (p) s.addImage({ path: p, x: 8.6, y: 0.5, w: 4.3, h: 2.6, sizing: { type: "cover", w: 4.3, h: 2.6 }, transparency: 15, objectName: "표지 그림" });
  notes(s, "세미나 소개 슬라이드. 발표 순서: 항공자력탐사 개요 → 드론 전용 프로그램이 필요한 이유 → DroneMag Studio 개발 내용 → 안동 사례.");
}
{
  const s = content("발표 순서");
  cards(s, [
    { head: "항공자력탐사 개요", body: "측정량과 자기이상, 처리 흐름, 측선 간격·고도와 해상도, 유인 항공과 드론의 차이", color: C.accent2 },
    { head: "드론 전용 프로그램의 필요성", body: "드론 고유의 잡음원, 기존 처리 흐름의 빈틈, 요구사항", color: C.accent1 },
    { head: "DroneMag Studio 개발 내용", body: "구조, 16단계 작업 흐름, 전처리·보정·레벨링·그리딩·파생 격자·해석, 지도와 참조 레이어, 품질 관리", color: C.accent4 },
    { head: "안동 적용 사례", body: "2026-10-28 MagArrow 드론 탐사 자료의 처리 결과와 기존 항공 자력도·시추 위치와의 비교", color: C.accent3 },
  ], { y: 1.9, h: 3.0, headSize: 16, bodySize: 13 });
}

// ============================================================ 2. overview
section("1. 항공자력탐사 개요");
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: secTitle });
  s.addText("1. 항공자력탐사 개요", { placeholder: "title" });
  s.addText("무엇을 재고, 무엇을 보정해서, 무엇을 보는가", { placeholder: "body" });
}
{
  const s = content("자력탐사의 측정량 — 총자기장과 자기이상");
  bullets(s, [
    { text: "총자기장 F: 지구자기장(약 50,000 nT)과 지하 자화체의 장을 합친 크기", bold: true },
    { text: "자기이상 ΔF = F − (지구 기준장 IGRF) − (시간 변화)", sub: true },
    { text: "자화체의 장은 쌍극자 형태: 한국(복각 약 53°)에서는 물체 바로 위가 아니라 남쪽에 양, 북쪽에 음의 이상", bold: true },
    { text: "극자화 변환(RTP)이 이상을 물체 바로 위로 옮겨 해석을 쉽게 한다", sub: true },
    { text: "강자성 광물(자철석)·철제 구조물·화성암체가 주요 이상원", bold: true },
    { text: "측정은 스칼라(세슘·루비듐 광펌핑)나 벡터(플럭스게이트) 자력계로", sub: true },
  ], { w: 12.1, h: 2.3, fontSize: 14 });
  picture(s, "concept_rtp.png", 1.0, 3.75, 11.3, 3.0, "자화체 하나의 이상(좌)과 RTP 후(중), 남북 단면(우) — 합성 자료");
  notes(s, "복각 53도에서 쌍극자 이상이 어떻게 생기는지, RTP가 왜 필요한지 설명.");
}
{
  const s = content("항공자력탐사 자료처리의 표준 흐름");
  const steps = ["자료 취득", "전처리", "측선 정리", "일변화 보정", "IGRF 보정", "레벨링", "그리딩", "파생 격자", "해석"];
  const subs = ["자력계·GPS·IMU", "스파이크·필터", "선회·이착륙 제거", "기준국·관측소", "지구 기준장 제거", "헤딩·교차점·통계", "최소곡률 등", "RTP·미분·AS", "심도·역산·표적"];
  const bw = 1.28, gap = 0.1, x0 = M + 0.05, y = 1.8;
  steps.forEach((t, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: bw, h: 1.7, fill: { color: i >= 5 ? C.accent2 : i >= 3 ? C.accent4 : C.accent5 }, line: { color: C.background1 }, rectRadius: 0.1, objectName: `단계 ${i + 1}` });
    s.addText([{ text: t + "\n", options: { bold: true, fontSize: 12, breakLine: true } }, { text: subs[i], options: { fontSize: 9.5 } }], { x: x + 0.05, y, w: bw - 0.1, h: 1.7, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: `단계 글 ${i + 1}` });
  });
  s.addText("회색: 취득·정리  |  녹색: 보정  |  파랑: 격자와 해석", { x: M, y: 3.7, w: 12, h: 0.35, fontSize: 12, color: C.accent5, isTextBox: true, margin: 0, objectName: "범례" });
  bullets(s, [
    "각 단계는 서로 다른 오차원을 겨냥한다: 시간 변화(일변화), 공간 기준(IGRF), 측선 간 불일치(레벨링), 표본화(그리딩)",
    "유인 항공에서는 수십 년간 정립된 흐름이며 상용 소프트웨어(Oasis montaj 등)가 이를 구현한다",
    "드론 탐사도 같은 뼈대를 쓰지만, 뒤에서 보듯 드론에만 있는 단계가 추가로 필요하다",
  ], { y: 4.2, w: 12, h: 2.6, fontSize: 14 });
}
{
  const s = content("측선 간격과 비행 고도가 해상도를 정한다");
  picture(s, "concept_sampling.png", M, 1.4, 12.1, 4.2, "같은 가상의 지하 장(좌)을 유인 항공(중)과 드론(우)이 본 모습 — 고도가 높을수록 작은 이상원은 사라지고, 측선이 성길수록 복원이 거칠어진다");
  bullets(s, [
    "포텐셜장은 고도 h에서 파장 λ 성분이 exp(−2πh/λ)로 줄어든다: 고도 300 m에서는 수십 m 규모 이상원이 거의 사라진다",
    "측선 간격의 2배보다 짧은 파장은 복원되지 않는다(샘플링 정리): 드론의 25 m 간격이 유인 항공의 400 m보다 16배 세밀",
  ], { y: 6.0, w: 12, h: 1.1, fontSize: 13 });
}
{
  const s = content("유인 항공 자력탐사 vs 드론 자력탐사");
  const rows = [
    [{ text: "항목", options: { bold: true, color: C.background1, fill: { color: C.accent2 } } }, { text: "유인 항공(고정익·헬기)", options: { bold: true, color: C.background1, fill: { color: C.accent2 } } }, { text: "드론(멀티콥터·VTOL)", options: { bold: true, color: C.background1, fill: { color: C.accent2 } } }],
    ["비행 고도", "80~300 m", "20~60 m (지형 추종)"],
    ["측선 간격", "100~500 m", "10~50 m"],
    ["하루 조사 면적", "수백 km²", "수 km²"],
    ["센서 탑재", "스팅어·버드, 보상 비행 필수", "기체 아래 매달림·거치, 센서 흔들림"],
    ["기준국", "별도 지상 기준국", "없는 경우 많음 → 관측소 자료 대체"],
    ["자료 특성", "긴 측선, 안정 고도", "짧은 측선·잦은 선회, 고도 변화, 헤딩 오차 큼"],
    ["처리 소프트웨어", "상용(Oasis montaj 등) 정착", "드론 단계를 다루는 도구 부족"],
  ];
  s.addTable(rows, { x: M, y: 1.5, w: W - 2 * M, colW: [2.2, 4.9, 5.0], fontSize: 13, color: C.text2, border: { type: "solid", color: "D9DEE5", pt: 0.75 }, rowH: 0.5, valign: "middle", objectName: "비교표" });
  s.addText("드론은 '작고 세밀하게, 낮고 자주'. 자료의 성격이 달라지므로 처리도 달라져야 한다.", { x: M, y: 5.9, w: 12, h: 0.5, fontSize: 15, bold: true, color: C.accent1, isTextBox: true, margin: 0, objectName: "요점" });
}

// ============================================================ 3. why a drone program
section("2. 드론 전용 프로그램의 필요성");
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: secTitle });
  s.addText("2. 드론 전용 프로그램의 필요성", { placeholder: "title" });
  s.addText("드론 자료에만 있는 잡음과, 기존 흐름이 비워 둔 자리", { placeholder: "body" });
}
{
  const s = content("드론 자력탐사 고유의 잡음원");
  picture(s, "concept_drone_noise.png", M, 1.35, 12.1, 3.6, "합성 예시 — 센서 흔들림(좌), 왕복 측선의 헤딩 오차가 만드는 줄무늬(중), 지형 추종 비행의 고도 변화(우)");
  cards(s, [
    { head: "센서 흔들림(스웨이)", body: "매달린 센서가 흔들리면 자세가 바뀌어 수 nT의 주기 잡음. IMU로 검출해 제외하거나 보정", color: C.accent1 },
    { head: "헤딩 오차", body: "센서 방향에 따른 읽기 편차. 왕복 측선마다 부호가 바뀌어 격자에 줄무늬(코러게이션)", color: C.accent3 },
    { head: "고도 변화", body: "지형 추종 비행은 고도가 수십 m 변하고, 같은 이상이 다른 진폭으로 기록된다", color: C.accent4 },
  ], { y: 5.5, h: 1.5, headSize: 13, bodySize: 11 });
}
{
  const s = content("기존 처리 흐름의 빈틈");
  cards(s, [
    { head: "줄무늬를 격자 뒤에서만 다룬다", body: "헤딩·흔들림은 자료 취득 단계의 현상인데, 기존 흐름은 그리딩 뒤 마이크로레벨링으로 결과만 다듬는다. 원인은 남는다.", color: C.accent1 },
    { head: "기준국이 없으면 막힌다", body: "소규모 드론 탐사는 지상 기준국을 못 두는 경우가 많다. 주변 관측소 자료를 가져와 가상 기준국을 만드는 단계가 없다.", color: C.accent3 },
    { head: "고도가 달라도 같은 값으로 본다", body: "지형 추종 비행의 고도 차이를 보정하는 '고도 정규화'가 표준 흐름에 없다.", color: C.accent4 },
    { head: "비용과 숙련도", body: "상용 소프트웨어는 고가이고 단계마다 수작업·전문 지식이 필요하다. 탐사 현장에서 바로 결과를 보기 어렵다.", color: C.accent2 },
    { head: "짧은 측선·잦은 선회", body: "수백 개의 짧은 측선, 이착륙·선회 구간, 반복 비행을 자동으로 가려내고 합치는 도구가 필요하다.", color: C.accent6 },
    { head: "해석까지 한 번에", body: "격자 생성에서 끝나지 않고 심도 추정·역산·표적 탐지까지 같은 화면에서 이어져야 현장에서 판단할 수 있다.", color: C.accent5 },
  ], { cols: 3, y: 1.5, h: 2.5, headSize: 14, bodySize: 12 });
}
{
  const s = content("DroneMag Studio의 설계 목표");
  bullets(s, [
    { text: "하나의 웹 화면에서 적재부터 해석·보고서까지", bold: true },
    { text: "값을 바꾸면 지도에 바로 반영, 각 단계의 판단 근거를 화면에 표시", sub: true },
    { text: "드론 고유 단계를 내장", bold: true },
    { text: "스웨이 검출, 헤딩 효과 캘리브레이션(선회 자료 활용), 헤딩 레벨링, 비행 고도 정규화, INTERMAGNET 가상 기준국", sub: true },
    { text: "자동 판단 + 사람의 수정", bold: true },
    { text: "측선 자동 판별을 지도에서 영역으로 고치고, 인공구조물 이상을 골라 제거", sub: true },
    { text: "재현 가능한 처리", bold: true },
    { text: "모든 기본값에 측정 근거, 처리 보고서 자동 생성, 프로젝트 저장·복원, 자동 테스트", sub: true },
    { text: "설치 한 번, 로컬 실행", bold: true },
    { text: "인터넷 없는 현장에서도 동작(오프라인 지도), 자료는 사용자 PC를 떠나지 않음", sub: true },
  ], { w: 7.2, fontSize: 14 });
  picture(s, "ui_grid.png", 8.0, 1.5, 4.8, 2.7, "DroneMag Studio 화면");
  picture(s, "andong_anomaly_on_imagery.png", 8.0, 4.5, 4.8, 2.5);
}

// ============================================================ 4. development
section("3. DroneMag Studio 개발 내용");
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: secTitle });
  s.addText("3. DroneMag Studio 개발 내용", { placeholder: "title" });
  s.addText("구조, 작업 흐름, 알고리듬, 품질 관리", { placeholder: "body" });
}
{
  const s = content("프로그램 구조");
  const box = (x, y, w, h, head, body, color) => {
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color }, line: { color }, rectRadius: 0.1, objectName: head });
    s.addText([{ text: head + "\n", options: { bold: true, fontSize: 15, breakLine: true } }, { text: body, options: { fontSize: 11.5 } }], { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h - 0.2, color: C.background1, valign: "top", isTextBox: true, margin: 0, objectName: head + " 글" });
  };
  box(M, 1.5, 3.8, 2.3, "프런트엔드 (React + Leaflet)", "16단계 작업 패널, 캔버스 지도, 격자 오버레이·등고선·힐쉐이드, 측선 편집, 3D 역산 뷰, 참조 레이어, 대화 도우미", C.accent2);
  box(4.75, 1.5, 3.8, 2.3, "백엔드 (FastAPI)", "프로젝트 상태 관리, 처리 파이프라인 호출, GeoTIFF·CSV·보고서 내보내기, 자동 저장, INTERMAGNET·지질도 WMS 연계", C.accent4);
  box(8.9, 1.5, 3.8, 2.3, "magcore (Python 라이브러리)", `처리·해석 알고리듬 ${n(62)}개 모듈, 약 ${n(15500)}줄. DroneMag Studio와 지뢰탐지용 MineMag가 함께 쓴다`, C.accent1);
  s.addShape(pres.ShapeType.line, { x: 4.4, y: 2.65, w: 0.35, h: 0, line: { color: C.accent5, width: 2, endArrowType: "triangle" }, objectName: "화살표 1" });
  s.addShape(pres.ShapeType.line, { x: 8.55, y: 2.65, w: 0.35, h: 0, line: { color: C.accent5, width: 2, endArrowType: "triangle" }, objectName: "화살표 2" });
  bullets(s, [
    "실행: run.bat 하나로 가상환경 준비·빌드·서버 실행·브라우저 열기. 설치본(Inno Setup)도 제공",
    `자동 테스트 ${n(706)}개(백엔드 90개 파일)와 MineMag ${n(58)}개가 모든 알고리듬의 기본값과 수치를 고정`,
    "기술 매뉴얼(이론·단계별 상세·기본값 근거·학습 노트)과 처리 보고서(Markdown)를 함께 유지",
    "입력: MagArrow·MagComp·일반 CSV 자동 인식, 베이스 로거, GeoTIFF·KML/KMZ·GeoJSON·CSV 참조 레이어, 대용량 정사영상",
  ], { y: 4.1, w: 12, h: 2.8, fontSize: 14 });
}
{
  const s = content("16단계 작업 흐름을 한 화면에");
  picture(s, "ui_grid.png", M, 1.4, 8.0, 4.5, "안동 자료를 복구해 격자와 등고선을 만든 화면");
  bullets(s, [
    "1 드론 자료 · 2 베이스 자료 · 3 처리 · 4 격자",
    "5 역산·오일러·디컨볼루션 · 5-1 고도 정규화",
    "6 표적 탐지 · 7 측선 편집 · 7-1 인공구조물 제거",
    "8 프로파일 · 9 반복 비행 비교 · 10 QC 인증",
    "11 배경지도·지질도 · 12 참조 레이어",
    "13 등고선·힐쉐이드 · 14 유망도 · 15 보고서",
    "16 오프라인 지도 · 대화 도우미",
  ], { x: 8.9, y: 1.45, w: 4.0, h: 5.2, fontSize: 12.5 });
}
{
  const s = content("자료 적재와 품질검사 (QC)");
  cards(s, [
    { head: "형식 자동 인식", body: "MagArrow 10 Hz CSV(끝 쉼표·사전 보정 컬럼 포함), MagComp, 일반 CSV. 여러 비행 파일을 시간순으로 합치고 중복 시각·비정상 좌표 제거", color: C.accent2 },
    { head: "GPS–자력계 시간지연", body: "속도와 이상의 상관으로 지연을 추정·보정. 지연이 남으면 왕복 측선이 어긋나 줄무늬가 된다", color: C.accent4 },
    { head: "잡음 QC", body: "4차·8차 차분 RMS로 측선별 잡음을 재고 기준(0.1 nT)을 넘는 측선을 표시. 표본율·속도·고도 QC", color: C.accent3 },
    { head: "반복 비행 재현성", body: "같은 측선을 다시 날았을 때의 차이로 처리 품질을 수치화(블록 평균 비교)", color: C.accent1 },
    { head: "베이스 QC", body: "기준국 자료의 공백·스파이크·시계 오프셋 검사와 보정", color: C.accent6 },
    { head: "줄무늬 정량 진단", body: "격자의 측선 방향 스펙트럼으로 줄무늬 비율을 재서 레벨링 효과를 숫자로 확인", color: C.accent5 },
  ], { cols: 3, y: 1.5, h: 2.5, headSize: 14, bodySize: 12 });
}
{
  const s = content("전처리와 측선 정리");
  bullets(s, [
    { text: "스파이크 제거: 이동 중앙값 + MAD(강건 통계). 급경사 구간은 창을 넓혀 실제 이상을 지우지 않음", bold: true },
    { text: `안동: ${n(procSum?.despike?.pct_spikes_removed, 1)}% (${n(procSum?.despike?.n_spikes_removed)}점) 제거`, sub: true },
    { text: "저역통과: Butterworth 영위상(기본 1 Hz) 또는 Savitzky–Golay·이동평균", bold: true },
    { text: "측선 자동 판별: 주 방향 기준 분류, 선회·이착륙·대기 구간 제외, 짧은 조각 병합", bold: true },
    { text: `안동: ${n(procSum?.n_lines)}개 측선, 자동 제외 ${n(procSum?.n_excluded_auto)}점`, sub: true },
    { text: "지도에서 수동 편집: 영역을 그려 제외/복원, 잘못 나뉜 측선 합치기", bold: true },
    { text: "스웨이 검출: IMU 가속도·자이로로 센서 흔들림 구간을 찾아 표시·제외", bold: true },
  ], { w: 6.6, fontSize: 14 });
  picture(s, "andong_profile.png", 7.4, 1.5, 5.4, 2.0, "한 측선의 처리 전후");
  picture(s, "andong_lines.png", 7.4, 3.8, 5.4, 3.2, "측선 판별 결과(원시 총자기장 색)");
}
{
  const s = content("일변화 보정 — 기준국이 없을 때 INTERMAGNET 가상 기준국");
  picture(s, "andong_base.png", M, 1.4, 12.1, 3.5, "안동 탐사 이틀간의 결합 기준국 시계열");
  const st = (summary.intermagnet && summary.intermagnet.stations) || [];
  const stText = st.length ? st.map((x) => `${x.iaga_code} ${n(x.distance_km)} km`).join(" · ") : "청양(CYG) · 가노야(KNY) · 가키오카(KAK) · 메모로(MMB)";
  bullets(s, [
    "조사지역 주변 동서남북에서 가장 가까운 관측소 4곳을 자동 선택하고 거리가중(IDW)으로 결합",
    `안동: ${stText} — 미게시 자료는 전후일로 추정, 관측소 교체 구간은 단차 제거`,
    "지상 기준국이 있으면 그 자료를 쓰고, 시계 오프셋·공백은 QC에서 보정. 둘을 비교하는 화면도 제공",
  ], { y: 5.3, w: 12, h: 1.8, fontSize: 13 });
}
{
  const s = content("드론 전용 보정 — 헤딩, 흔들림, 고도");
  cards(s, [
    { head: "헤딩 효과 캘리브레이션", body: "센서 방향(θ, φ)에 따른 읽기 편차를 캘리브레이션 비행 또는 측선의 선회 구간에서 추정해 원자료 단계에서 보정(Geometrics MagArrow 제조사 방법). 교차 검증으로 품질 판정", color: C.accent1 },
    { head: "헤딩 레벨링", body: "왕복 측선의 국소 평면 적합으로 방향 오프셋을 측정. 오프셋이 산포보다 작으면 '헤딩 효과로 볼 수 없다'며 적용하지 않는 보수적 규칙", color: C.accent3 },
    { head: "비행 고도 정규화", body: "지형 추종 비행의 고도 차이를 3층 등가층(boosting)으로 기준 고도의 값으로 환산. 해남 자료: 고도 산포 47.6 nT → 잔차 4.9 nT, 83초", color: C.accent4 },
  ], { y: 1.5, h: 2.6, headSize: 14, bodySize: 12 });
  bullets(s, [
    `안동 헤딩 레벨링: 오프셋 ${n(procSum?.heading_correction?.offset_nt, 1)} nT, 구역별 산포 ${n(procSum?.heading_correction?.offset_spread_nt, 1)} nT → 적용 보류(산포 > 오프셋). 대신 통계 레벨링이 거칠기 ${n(procSum?.statistical_leveling?.roughness_before_nt, 1)} → ${n(procSum?.statistical_leveling?.roughness_after_nt, 1)} nT로 줄임`,
    "스웨이·헤딩·고도는 모두 '격자 전' 단계: 원인이 있는 자리에서 고치고, 마이크로레벨링은 마지막 보조 수단으로만",
  ], { y: 4.4, w: 12, h: 2.5, fontSize: 13.5 });
}
{
  const s = content("레벨링과 그리딩");
  bullets(s, [
    { text: "세 가지 레벨링, 세 가지 문제", bold: true },
    { text: "교차점(타이라인) 레벨링: 타이라인이 있을 때 교차점 차이를 최소화", sub: true },
    { text: "통계 레벨링: 이웃 측선 간 저주파 차이를 추세창으로 제거 — 타이라인 없는 드론 탐사의 기본", sub: true },
    { text: "마이크로레벨링(디코러게이션): 격자에서 측선 방향 파장을 선택적으로 억제", sub: true },
    { text: "그리딩", bold: true },
    { text: "블록 평균 → 최소곡률(Briggs)·선형·큐빅·스플라인·boxing. 측선 방향 평활화로 측선/횡측선 표본 밀도 차이를 보정", sub: true },
    { text: "덮임 공백은 측선 간격의 국소 추정으로 채우되 볼록 껍질 밖으로는 외삽하지 않음", sub: true },
    { text: "격자 신뢰도 레이어: 측점 밀도·거리로 각 셀의 믿음직함을 표시", sub: true },
  ], { w: 6.3, fontSize: 13.5 });
  picture(s, "andong_anomaly.png", 7.1, 1.4, 5.8, 5.2, "안동 총자기장 이상 격자(셀 5 m)와 등고선");
}
{
  const s = content("파생 격자와 변환");
  picture(s, "andong_derivatives.png", M, 1.3, 12.1, 4.35, "안동 자료의 RTP · 해석신호 · 총수평미분 · 1차 수직미분 · 틸트각 · 상향연속 50 m");
  bullets(s, [
    "FFT 기반 변환 전에 측선 간격이 정하는 해상도 한계로 저역통과하고, 가장자리는 등가소스로 연속해 테두리 왜곡을 없앤다",
    "RTP/RTE, 1·2차 수직미분, 수평미분과 조합량(AS·THDR·틸트·θ), 상향·하향 연속, 추세 제거, 마이크로레벨링. 값·컬러 GeoTIFF로 내보내기",
  ], { y: 6.15, w: 12, h: 1.0, fontSize: 13 });
}
{
  const s = content("해석 산출물");
  cards(s, [
    { head: "심도 추정", body: "방사평균 파워스펙트럼, 오일러 디컨볼루션(구조지수 선택, 창 이동, 불확실도 필터), 쌍극자 피팅", color: C.accent2 },
    { head: "3차원 자화율 역산", body: "심도가중 Tikhonov + IRLS focusing, 지형 반영 활성 셀, 지질 단위별 참조 모델, 단면·볼륨 뷰", color: C.accent1 },
    { head: "구조 해석", body: "다중스케일 경계, 선구조 추출, 접촉면 해석, 지질 단위별 이상 통계, 유망도 매핑", color: C.accent4 },
    { head: "근지표 표적 탐지", body: "국지 극값 후보 → 단일 유도자화 쌍극자 비선형 적합 → 심도·모멘트·품질 등급. 지뢰·불발탄용 MineMag로 확장", color: C.accent3 },
  ], { cols: 2, y: 1.5, h: 1.9, headSize: 14, bodySize: 12, w: 5.9 });
  bullets(s, [
    "모든 해석 결과는 지도 위에 겹쳐 보이고, CSV·Shapefile·GeoJSON·GeoTIFF·메쉬(.npz)로 내보낸다",
    "대화 도우미: 실제 프로젝트 값을 조회(지점 조회, 통계)해 답하는 Claude API 기반 해석 보조",
  ], { y: 5.75, w: 12, h: 1.2, fontSize: 13.5 });
}
{
  const s = content("지도, 참조 레이어, 현장 운용");
  picture(s, "ui_layers.png", M, 1.4, 7.6, 4.3, "안동 정사영상(114 GB GeoTIFF를 디스크에서 바로 타일로)과 시추 위치(KMZ)를 격자 위에 겹친 화면");
  bullets(s, [
    { text: "배경지도: OSM·Esri·Google 위성, 오프라인 타일 미리 받기", bold: true },
    { text: "KIGAM 지질도 WMS(1:5만·25만·100만, 단층)를 타일처럼 캐시", sub: true },
    { text: "참조 레이어", bold: true },
    { text: "GeoTIFF 업로드, 대용량 GeoTIFF 경로 연동(요청 타일만 읽기), 타일 폴더", sub: true },
    { text: "KML/KMZ·GeoJSON·CSV 점·선·면(시추공, 시료, 경계)", sub: true },
    { text: "격자 표시: 컬러맵·스트레치, 힐쉐이드, 등고선(항상 계산, 표시만 토글)", bold: true },
    { text: "프로젝트 저장·자동 저장·복구, 처리 보고서(.md)", bold: true },
  ], { x: 8.5, y: 1.45, w: 4.4, h: 5.3, fontSize: 12.5 });
}
{
  const s = content("품질 관리와 재현성");
  stat(s, M, 1.6, 2.8, n(706), "자동 테스트 (DroneMag Studio 백엔드)", C.accent2);
  stat(s, M + 3.1, 1.6, 2.8, n(62), "magcore 처리·해석 모듈", C.accent4);
  stat(s, M + 6.2, 1.6, 2.8, "16", "작업 단계(패널)", C.accent3);
  stat(s, M + 9.3, 1.6, 2.8, "155+", "커밋 (2026-07 ~ 11)", C.accent1);
  bullets(s, [
    { text: "기본값마다 측정 근거를 남긴다", bold: true },
    { text: "예: 미분 전 평활화 계수 1.0(인공 줄무늬 최소), 통계 레벨링 추세창 9측선, 헤딩 보정 신뢰 규칙(산포 < 오프셋), 격자 덮임 거리 = 측선 간격의 국소 추정", sub: true },
    { text: "모든 단계가 화면에 수치와 경고를 보여 준다", bold: true },
    { text: "예: '구역별 오프셋 산포가 오프셋보다 커서 헤딩 효과로 볼 수 없어 적용하지 않았습니다'", sub: true },
    { text: "처리 보고서(.md)와 QC 인증서가 어떤 값으로 무엇을 했는지 기록", bold: true },
    { text: "기술 매뉴얼: 이론적 배경, 단계별 상세, 기본값과 근거, 알려진 한계, 학습 노트", bold: true },
  ], { y: 3.4, w: 12, h: 3.5, fontSize: 14 });
}
{
  const s = content("파생 제품 — MineMag (지뢰·불발탄 근지표 표적 탐지)");
  picture(s, "landmine_grids.png", M, 1.4, 7.4, 4.9, "KIGAM 시험장: 드론 0.5 m 비행과 사족보행로봇 2회 반복 측정의 격자");
  bullets(s, [
    "같은 magcore 위에 세운 전용 프로그램: MagDrone R3(플럭스게이트 2탐침, 200 Hz)와 광펌핑 스칼라 쌍 자료",
    "플랫폼 잡음 QC(보행 주기 스펙트럼, 탐침 상관), 논문 워크플로(리샘플·Hampel·가우시안 LPF·선회 제외·센서별 헤딩 보정)",
    "0.125 m 격자, THD/ASA 극대점 피킹(K×MAD), 측정 간 매칭·등급, 매설 심도, 자화 방향 판정",
    "영역 지정·저장, 배경지도·오프라인 타일, 정사영상 — DroneMag Studio의 지도 기능을 공유",
  ], { x: 8.3, y: 1.45, w: 4.6, h: 5.3, fontSize: 12.5 });
}

// ============================================================ 5. Andong case
section("4. 안동 적용 사례");
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: secTitle });
  s.addText("4. 안동 적용 사례", { placeholder: "title" });
  s.addText("2026년 10월 28일 드론 자력탐사 — 처리와 기존 항공 자력도·시추 위치 비교", { placeholder: "body" });
}
{
  const s = content("조사 개요");
  picture(s, "andong_lines.png", M, 1.4, 6.3, 5.15, "비행 측선과 원시 총자기장 (Esri 위성영상 위)");
  const d = summary.drone || {}, p = summary.process || {};
  stat(s, 7.3, 1.5, 2.7, n(p.n_points || d.n_points), "측정점 (10 Hz)", C.accent2);
  stat(s, 10.2, 1.5, 2.7, n(p.n_lines), "자동 판별 측선", C.accent4);
  stat(s, 7.3, 3.2, 2.7, n(summary.n_files), "비행 파일", C.accent3);
  stat(s, 10.2, 3.2, 2.7, n(p.line_spacing_m, 0) + " m", "추정 측선 간격", C.accent1);
  bullets(s, [
    "장비: Geometrics MagArrow(세슘 스칼라, 1,000 Hz → 10 Hz), DJI 기체, 안동시 서부 (36.53°N, 128.48°E)",
    "기존 자료: 항공 자력탐사 RTP 도면과 1:5만 지질도(그림을 EPSG:5187로 지오레퍼런싱해 참조 레이어로 사용)",
    "시추 위치 2곳(Google Earth KMZ)을 참조 레이어로 겹침",
  ], { x: 7.3, y: 4.9, w: 5.6, h: 2.1, fontSize: 12.5 });
}
{
  const s = content("처리 결과 요약");
  const hc = procSum.heading_correction || {}, sl = procSum.statistical_leveling || {}, dq = procSum.diurnal || {}, nq = procSum.noise_qc || {};
  stat(s, M, 1.6, 2.8, n(procSum?.despike?.pct_spikes_removed, 1) + "%", "스파이크 제거 비율", C.accent1);
  stat(s, M + 3.1, 1.6, 2.8, n(dq.coverage_pct, 0) + "%", "일변화 보정 커버리지 (가상 기준국)", C.accent4);
  stat(s, M + 6.2, 1.6, 2.8, `${n(sl.roughness_before_nt, 1)} → ${n(sl.roughness_after_nt, 1)}`, "통계 레벨링 전후 측선 간 거칠기 (nT)", C.accent2);
  stat(s, M + 9.3, 1.6, 2.8, n(nq.overall_rms_4th_diff_nt, 2) + " nT", "잡음 QC: 4차 차분 RMS", C.accent3);
  const rows = [
    [{ text: "단계", options: { bold: true, color: C.background1, fill: { color: C.accent2 } } }, { text: "설정", options: { bold: true, color: C.background1, fill: { color: C.accent2 } } }, { text: "결과", options: { bold: true, color: C.background1, fill: { color: C.accent2 } } }],
    ["일변화", "INTERMAGNET 4관측소 IDW 결합", `${n(dq.coverage_pct, 0)}% 덮음, 외삽 ${n(dq.n_extrapolated)}점`],
    ["IGRF", `복각 ${n(procSum.inclination_deg, 1)}°, 편각 ${n(procSum.declination_deg, 1)}°`, "지구 기준장 제거 → 자기이상"],
    ["헤딩 레벨링", "국소 평면, 조용한 구간 40%", `오프셋 ${n(hc.offset_nt, 1)} nT, 산포 ${n(hc.offset_spread_nt, 1)} nT → ${hc.applied ? "적용" : "보류"}`],
    ["통계 레벨링", `추세창 ${n(sl.trend_window_lines)}측선`, `최대 보정 ${n(sl.max_shift_nt, 1)} nT, RMS ${n(sl.rms_shift_nt, 1)} nT`],
    ["그리딩", "셀 5 m, 최소곡률, 평면 추세", `격자 ${(summary.grids && summary.grids.grid && summary.grids.grid.shape || []).join(" × ")} 셀`],
  ];
  s.addTable(rows, { x: M, y: 3.5, w: W - 2 * M, colW: [2.2, 4.6, 5.3], fontSize: 12.5, color: C.text2, border: { type: "solid", color: "D9DEE5", pt: 0.75 }, rowH: 0.48, valign: "middle", objectName: "처리표" });
}
{
  const s = content("자기이상 격자");
  picture(s, "andong_anomaly.png", M, 1.35, 6.6, 5.25, "총자기장 이상, 셀 5 m, 등고선");
  picture(s, "andong_anomaly_on_imagery.png", 7.4, 1.35, 5.5, 5.25, "위성영상·시추 위치 위에 겹친 모습");
}
{
  const s = content("파생 격자");
  picture(s, "andong_derivatives.png", M, 1.35, 12.1, 5.7);
}
{
  const s = content("기존 항공 자력탐사와의 비교");
  picture(s, "andong_airborne_vs_drone.png", M, 1.3, 12.1, 4.55, "같은 영역: 기존 항공 RTP(좌)와 드론 RTP(우), 노란 점은 시추 위치");
  bullets(s, [
    "항공 자력도의 큰 양의 이상대 서쪽 끝에서 드론 자료는 수백 m 규모의 세부 구조를 분해한다",
    "시추 위치 2곳은 드론 RTP의 고이상대 가장자리에 놓인다 — 세부 구조와 시추 결과의 대조가 다음 단계",
  ], { y: 6.3, w: 12, h: 0.95, fontSize: 13 });
}
{
  const s = content("심도 추정 — 오일러 디컨볼루션");
  picture(s, "andong_euler.png", M, 1.35, 6.6, 5.25, "오일러 해(구조지수 1, 창 100 m)를 자기이상(회색) 위에 표시, 색은 심도");
  const e = summary.euler || {};
  bullets(s, [
    `해 ${n(e.n_solutions)}개, 심도 중앙값 ${n(e.depth_median_m, 0)} m`,
    "구조지수 0~3을 바꿔 가며 접촉면·암맥·원통·구형 이상원을 가정해 볼 수 있고, 불확실도가 큰 해는 자동으로 거른다",
    "방사평균 파워스펙트럼과 3차원 역산으로 교차 확인 — 역산은 지질도 단위별 참조 모델을 받는다",
  ], { x: 7.4, y: 1.45, w: 5.5, h: 5.3, fontSize: 13.5 });
}

// ============================================================ 6. closing
section("맺음");
{
  const s = content("정리와 다음 단계");
  cards(s, [
    { head: "정리", body: "드론 자력탐사는 유인 항공의 처리 뼈대에 드론 고유 단계(스웨이·헤딩·고도·가상 기준국)를 더해야 한다. DroneMag Studio는 이를 한 화면의 재현 가능한 흐름으로 묶었다.", color: C.accent2 },
    { head: "검증", body: "해남·안동·KIGAM 시험장 자료로 단계별 수치를 확인했고, 706개 자동 테스트가 기본값을 고정한다.", color: C.accent4 },
    { head: "다음 단계", body: "안동 시추 결과와의 대조, 기술이전(저고도 광역 정밀 드론 자력탐사 자료 취득·처리 기술), MineMag 현장 적용, 고도 정규화·자동 보고서 고도화", color: C.accent1 },
  ], { y: 1.6, h: 2.9, headSize: 16, bodySize: 13 });
  bullets(s, [
    "소스와 매뉴얼: C:\\magnetic (DroneMag Studio, MineMag, magcore, docs/TECHNICAL_MANUAL.md)",
    "문의: 방은석, 한국지질자원연구원",
  ], { y: 4.9, w: 12, h: 1.5, fontSize: 13 });
}
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: secTitle });
  s.addText("감사합니다", { placeholder: "title" });
  s.addText("질문과 토론", { placeholder: "body" });
}

const OUT = path.join(__dirname, "DroneMag_Studio_세미나.pptx");
pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
});
