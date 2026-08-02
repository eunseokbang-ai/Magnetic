"""Generates a human-readable Markdown processing report summarizing every
step actually applied to a project - what corrections ran, their key
numbers, and (if computed) grid/inversion/Euler results. Markdown rather
than PDF: no extra native-dependency PDF library needed, it's portable
(opens in any text editor, renders on GitHub), and any user who wants a
PDF can convert it in one step (browser print, pandoc, etc)."""
from __future__ import annotations

from datetime import datetime, timezone


def _fmt(value, digits=2, unit="") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def generate_report_markdown(
    drone_summary: dict | None,
    base_summary: dict | None,
    process_summary: dict | None,
    last_params: dict | None,
    inversion_summary: dict | None,
    euler_summary: dict | None,
    target_summary: dict | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# 드론 자력탐사 자료 처리 보고서")
    lines.append("")
    lines.append(f"생성 시각: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("## 1. 원본 자료")
    lines.append("")
    if drone_summary:
        lines.append(f"- 드론 자력 포인트 수: {drone_summary.get('n_points')}")
        tr = drone_summary.get("time_range") or [None, None]
        lines.append(f"- 비행 시간범위: {tr[0]} ~ {tr[1]}")
        mr = drone_summary.get("mag_range") or [None, None]
        lines.append(f"- 원시 자력값 범위: {_fmt(mr[0])} ~ {_fmt(mr[1])} nT")
    else:
        lines.append("- 드론 자료 없음")
    lines.append("")
    if base_summary:
        lines.append(f"- 베이스(일변화) 포인트 수: {base_summary.get('n_points')}")
        tr = base_summary.get("time_range") or [None, None]
        lines.append(f"- 베이스 시간범위: {tr[0]} ~ {tr[1]}")
    else:
        lines.append("- 베이스 자료 없음")
    lines.append("")

    if not process_summary:
        lines.append("_자료 처리가 아직 실행되지 않았습니다._")
        return "\n".join(lines) + "\n"

    lines.append("## 2. 처리 파라미터")
    lines.append("")
    lines.append("```json")
    import json

    lines.append(json.dumps(last_params or {}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## 3. 적용된 보정")
    lines.append("")

    gl = process_summary.get("gps_mag_lag")
    if gl and gl.get("lag_seconds"):
        lines.append(f"- **GPS-자력계 시간 오프셋 보정**: {gl['lag_seconds']}초 적용 (경계 {gl['n_points_dropped']}개 포인트 제외)")
    else:
        lines.append("- GPS-자력계 시간 오프셋 보정: 미적용")

    dsp = process_summary.get("despike")
    if dsp and dsp.get("enabled"):
        lines.append(f"- **스파이크 제거**: {dsp['n_spikes_removed']}개 제거 ({_fmt(dsp['pct_spikes_removed'])}%)")
    else:
        lines.append("- 스파이크 제거: 미적용")

    hec = process_summary.get("heading_effect_calibration")
    if hec and hec.get("enabled") and hec.get("applied"):
        source_label = "업로드된 캘리브레이션 비행" if hec.get("calibration_source") == "uploaded_file" else "측선 턴 구간 자동 추출"
        q = hec.get("quality_check") or {}
        quality_label = "PASS" if hec.get("quality_pass") else "FAIL (신뢰도 낮음)"
        lines.append(
            f"- **헤딩효과 캘리브레이션 보정** (Zhang et al. 2022, 자료 출처: {source_label}): "
            f"캘리브레이션 {hec.get('n_calibration_points')}개 포인트로 측선 {hec.get('n_survey_points_corrected')}개 포인트 보정 "
            f"({_fmt(hec.get('pct_survey_points_corrected'))}%, 평균 보정량 {_fmt(hec.get('mean_abs_correction_nt'))} nT, "
            f"외삽 {hec.get('n_survey_points_extrapolated')}개)"
        )
        if q.get("available"):
            lines.append(
                f"  - 캘리브레이션 품질 검증(교차검증): {quality_label} — 잔차 표준편차 {_fmt(q.get('residual_std_nt'))} nT "
                f"(기준값 {_fmt(hec.get('quality_threshold_nt'))} nT, peak-to-peak {_fmt(q.get('residual_p2p_nt'))} nT)"
            )
        if hec.get("coverage_warning"):
            lines.append(f"  - ⚠ {hec['coverage_warning']}")
    elif hec and hec.get("enabled") and hec.get("reason"):
        lines.append(f"- 헤딩효과 캘리브레이션 보정: 미적용 ({hec['reason']})")
    elif hec and hec.get("enabled"):
        lines.append("- 헤딩효과 캘리브레이션 보정: 미적용 (캘리브레이션 자료 없음)")
    else:
        lines.append("- 헤딩효과 캘리브레이션 보정: 미적용")

    sw = process_summary.get("sway_detection")
    if sw and sw.get("enabled") and sw.get("available"):
        lines.append(
            f"- **IMU 흔들림(스웨이) 검출**: {sw.get('n_points_excluded')}개 포인트 제외 "
            f"({_fmt(sw.get('pct_points_flagged'))}%, 신호: {sw.get('signal_used')})"
        )
    elif sw and sw.get("enabled"):
        lines.append("- IMU 흔들림(스웨이) 검출: 사용 설정됨이나 원본 파일에 자이로/가속도 데이터 없음 (미적용)")
    else:
        lines.append("- IMU 흔들림(스웨이) 검출: 미적용")

    diurnal = process_summary.get("diurnal")
    if diurnal:
        overlap_note = "정상 (비행-베이스 시간 겹침)" if diurnal.get("has_overlap") else "⚠ 베이스와 비행 시간이 겹치지 않음"
        lines.append(f"- **일변화 보정**: {overlap_note}, 기준값 {_fmt(diurnal.get('base_reference_value'))} nT")

    hc = process_summary.get("heading_correction")
    if hc and hc.get("applied"):
        lines.append(f"- **헤딩(비행방향) 보정**: 오프셋 {_fmt(hc.get('offset_nt'))} nT (매칭 {hc.get('n_matched_pairs')}쌍 중 조용한 {hc.get('n_quiet_pairs')}쌍 사용)")
    elif hc:
        lines.append(f"- 헤딩 보정: 미적용 ({hc.get('reason') or '해당 없음'})")

    cl = process_summary.get("crossover_leveling")
    if cl and cl.get("applied"):
        lines.append(
            f"- **타이라인(교차점) 보정**: 측선 {cl.get('n_survey_lines_corrected')}개 보정, "
            f"교차점 {cl.get('n_crossovers')}개, RMS {_fmt(cl.get('rms_before_nt'))} → {_fmt(cl.get('rms_after_nt'))} nT"
        )
    elif cl:
        lines.append(f"- 타이라인 보정: 미적용 ({cl.get('reason') or '해당 없음'})")
    lines.append("")

    lines.append("## 4. 측선 판별 결과")
    lines.append("")
    lines.append(f"- 총 포인트 수: {process_summary.get('n_points')}")
    lines.append(f"- 검출된 측선 수: {process_summary.get('n_lines')}")
    lines.append(f"- 유효(사용) 포인트 수: {process_summary.get('n_kept')}")
    lines.append(f"- 자동 제외 포인트 수: {process_summary.get('n_excluded_auto')}")
    lines.append(f"- 수동 포함/제외: +{process_summary.get('n_manual_included')} / -{process_summary.get('n_manual_excluded')}")
    lines.append(f"- 주 측선 방향: {_fmt(process_summary.get('dominant_azimuth_deg'))}°")
    lines.append(f"- 측선 간격: {_fmt(process_summary.get('line_spacing_m'))} m")
    lines.append(f"- 복각(Inclination): {_fmt(process_summary.get('inclination_deg'))}°, 편각(Declination): {_fmt(process_summary.get('declination_deg'))}°")
    lines.append("")

    lines.append("## 5. 자력값 통계 (유효 포인트 기준)")
    lines.append("")
    for label, key in [("자력 이상 (nT)", "anomaly_stats"), ("TMI (nT)", "tmi_stats")]:
        stats = process_summary.get(key)
        if stats:
            lines.append(f"- **{label}**: min {_fmt(stats.get('min'))}, max {_fmt(stats.get('max'))}, mean {_fmt(stats.get('mean'))}, std {_fmt(stats.get('std'))}")
    lines.append("")

    line_rows = process_summary.get("lines") or []
    if line_rows:
        lines.append("## 6. 측선별 요약")
        lines.append("")
        lines.append("| 측선 | 포인트 수 | 길이(m) | 방향그룹 | 헤딩보정(nT) |")
        lines.append("|---|---|---|---|---|")
        for l in line_rows:
            lines.append(
                f"| #{l.get('line_id')} | {l.get('n_points')} | {_fmt(l.get('length_m'), 0)} | "
                f"{l.get('heading_group') or '-'} | {_fmt(l.get('heading_shift_nt'))} |"
            )
        lines.append("")

    if inversion_summary:
        lines.append("## 7. 3차원 역산 결과")
        lines.append("")
        lines.append(f"- 관측점 수: {inversion_summary.get('n_obs')}, 활성 셀 수: {inversion_summary.get('n_active_cells')}")
        lines.append(f"- RMS 오차: {_fmt(inversion_summary.get('rms_misfit_nt'))} nT")
        lines.append(f"- 셀 크기: {_fmt(inversion_summary.get('cell_size_m'))} m, 레이어 수: {inversion_summary.get('n_layers')}")
        lines.append(f"- 탐사 심도: {_fmt(inversion_summary.get('depth_extent_m'))} m")
        if inversion_summary.get("resolution_warning"):
            lines.append(f"- ⚠ {inversion_summary['resolution_warning']}")
        lines.append("")

    if euler_summary:
        lines.append("## 8. 오일러 디컨볼루션 결과")
        lines.append("")
        lines.append(f"- 구조지수: {euler_summary.get('structural_index')}")
        lines.append(f"- 해의 개수: {euler_summary.get('n_solutions')}")
        ds = euler_summary.get("depth_stats")
        if ds:
            lines.append(f"- 추정 심도 범위: {_fmt(ds.get('min'))} ~ {_fmt(ds.get('max'))} m (평균 {_fmt(ds.get('mean'))} m)")
        lines.append("")

    if target_summary:
        lines.append("## 9. 근지표 표적탐지 결과 (쌍극자 피팅)")
        lines.append("")
        lines.append(
            "⚠ 쌍극자 모멘트는 상대적인 철질량 크기 등급일 뿐이며, 자력탐사만으로 표적의 종류(지뢰/포탄/전차 등)를 "
            "특정할 수 없습니다. 최소금속 물체는 신호가 거의 없어 탐지되지 않을 수 있습니다. 실제 위치 확인·처리는 "
            "반드시 전문 인력이 현장에서 검증해야 합니다."
        )
        lines.append("")
        lines.append(f"- 탐지된 표적 후보 수: {target_summary.get('n_targets')}")
        lines.append(f"- 진폭 임계값: {_fmt(target_summary.get('amplitude_threshold_nt'))} nT, 탐지 격자 크기: {_fmt(target_summary.get('cell_size_m'))} m")
        targets = target_summary.get("targets") or []
        if targets:
            lines.append("")
            lines.append("| 위도 | 경도 | 심도(m) | 쌍극자모멘트(A·m²) | 크기등급 | 첨두이상(nT) | 적합도 |")
            lines.append("|---|---|---|---|---|---|---|")
            for t in targets:
                lines.append(
                    f"| {_fmt(t.get('lat'), 5)} | {_fmt(t.get('lon'), 5)} | {_fmt(t.get('depth_m'))} | "
                    f"{_fmt(t.get('moment_am2'))} | {t.get('size_class')} | {_fmt(t.get('peak_anomaly_nt'))} | "
                    f"{_fmt(t.get('fit_quality'))} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"
