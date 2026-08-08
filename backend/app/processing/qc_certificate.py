"""Standard QC pass/fail certificate: bundles the individual QC indicators
this app already computes at various pipeline stages (noise QC, sampling
distance QC, file-to-file level offset check, heading-effect calibration
cross-validation, repeatability analysis, and how much data survived
automatic exclusion) into one PASS/FAIL-per-criterion certificate against
user-configurable acceptance thresholds, plus an overall PASS/FAIL.

Each criterion is independent and degrades gracefully to "평가 불가"
(not_evaluated) when its underlying metric isn't available (e.g.
repeatability analysis was never run, or heading calibration wasn't
enabled) - a missing metric is reported as such, never silently counted as
a pass or a fail.
"""
from __future__ import annotations


def _criterion(name: str, unit: str, value, threshold, detail: str) -> dict:
    """A "value must not exceed threshold" criterion. Criteria with a
    different pass condition (a bool or categorical flag rather than a
    numeric ceiling) are built as plain dicts inline below instead."""
    if value is None:
        return {"name": name, "unit": unit, "value": None, "threshold": threshold, "status": "not_evaluated", "detail": detail}
    return {
        "name": name,
        "unit": unit,
        "value": value,
        "threshold": threshold,
        "status": "pass" if value <= threshold else "fail",
        "detail": detail,
    }


def evaluate_qc_certificate(
    process_summary: dict,
    repeatability_summary: dict | None,
    noise_threshold_multiplier: float = 2.0,
    max_repeatability_1sigma_nt: float = 5.0,
    max_sampling_gap_pct: float = 5.0,
    max_excluded_pct: float = 30.0,
) -> dict:
    """Evaluate every applicable criterion against process_summary (the
    dict returned by store.py's Project.process_summary()) and, optionally,
    repeatability_summary (Project.repeatability_summary_cache - only
    populated if the user separately ran that analysis). Thresholds are
    exposed as parameters (backed by QcCertificateRequest in models.py) so
    a project's acceptance criteria can be tuned to the survey spec being
    delivered against, rather than a single hardcoded standard."""
    criteria = []

    noise = process_summary.get("noise_qc") or {}
    if noise.get("available") and noise.get("reference_threshold_4th_diff_nt"):
        noise_threshold = noise_threshold_multiplier * noise["reference_threshold_4th_diff_nt"]
        criteria.append(
            _criterion(
                "자력 노이즈 수준 (4차 차분 RMS)",
                "nT",
                noise["overall_rms_4th_diff_nt"],
                noise_threshold,
                f"샘플링 주파수({noise.get('sample_rate_hz'):.1f}Hz 기준) 참고 임계값의 {noise_threshold_multiplier:.1f}배 이내인지 확인 "
                f"(노이즈가 유독 심한 측선 {noise.get('n_lines_flagged', 0)}개 개별 플래그됨)",
            )
        )
    else:
        criteria.append(_criterion("자력 노이즈 수준 (4차 차분 RMS)", "nT", None, None, "노이즈 QC를 계산할 자료가 부족합니다."))

    if repeatability_summary and repeatability_summary.get("available"):
        criteria.append(
            _criterion(
                "반복측선 재현성 (1-sigma)",
                "nT",
                repeatability_summary["overall_noise_1sigma_nt"],
                max_repeatability_1sigma_nt,
                f"같은 구간을 반복 비행한 {repeatability_summary.get('n_groups', 0)}개 그룹의 평균 1-sigma 재현오차",
            )
        )
    else:
        criteria.append(
            _criterion("반복측선 재현성 (1-sigma)", "nT", None, None, "반복측선 분석을 아직 실행하지 않았거나 반복 구간이 없습니다.")
        )

    hec = process_summary.get("heading_effect_calibration") or {}
    if hec.get("enabled") and hec.get("applied") and hec.get("quality_check", {}).get("available"):
        criteria.append(
            {
                "name": "헤딩효과 캘리브레이션 교차검증",
                "unit": "",
                "value": "PASS" if hec["quality_pass"] else "FAIL",
                "threshold": "PASS",
                "status": "pass" if hec["quality_pass"] else "fail",
                "detail": f"held-out 잔차 표준편차 {hec['quality_check'].get('residual_std_nt', float('nan')):.2f}nT",
            }
        )
    else:
        criteria.append(
            {
                "name": "헤딩효과 캘리브레이션 교차검증",
                "unit": "",
                "value": None,
                "threshold": "PASS",
                "status": "not_evaluated",
                "detail": "헤딩효과 캘리브레이션이 켜져 있지 않거나 적용되지 않았습니다.",
            }
        )

    sampling = process_summary.get("sampling_qc") or {}
    if sampling.get("available"):
        criteria.append(
            _criterion(
                "샘플링 간격 계약기준 초과 비율",
                "%",
                sampling["pct_gaps_exceeding_tolerance"],
                max_sampling_gap_pct,
                f"허용거리 {sampling.get('gap_tolerance_m')}m를 넘는 연속 포인트 간격 {sampling.get('n_gaps_exceeding_tolerance', 0)}건",
            )
        )
    else:
        criteria.append(_criterion("샘플링 간격 계약기준 초과 비율", "%", None, None, "샘플링 QC를 계산할 자료가 부족합니다."))

    file_level = process_summary.get("file_level_check") or {}
    if file_level.get("available"):
        criteria.append(
            {
                "name": "파일간 DC 레벨 오프셋",
                "unit": "",
                "value": "이상 없음" if not file_level["flagged_any"] else "이상 감지됨",
                "threshold": "이상 없음",
                "status": "pass" if not file_level["flagged_any"] else "fail",
                "detail": f"{file_level.get('n_files', 0)}개 파일 비교, 임계값 {file_level.get('flag_threshold_nt', 0):.2f}nT",
            }
        )
    else:
        criteria.append(
            {
                "name": "파일간 DC 레벨 오프셋",
                "unit": "",
                "value": None,
                "threshold": "이상 없음",
                "status": "not_evaluated",
                "detail": "파일이 1개뿐이거나 파일 구분 정보가 없어 비교할 수 없습니다.",
            }
        )

    n_kept = process_summary.get("n_kept")
    n_excluded = process_summary.get("n_excluded_auto")
    if n_kept is not None and n_excluded is not None and (n_kept + n_excluded) > 0:
        excluded_pct = 100.0 * n_excluded / (n_kept + n_excluded)
        criteria.append(
            _criterion(
                "자동 제외 자료 비율",
                "%",
                excluded_pct,
                max_excluded_pct,
                f"이착륙/터닝/스웨이/중복측선 등으로 자동 제외된 포인트 {n_excluded:,}개 / 전체 {n_kept + n_excluded:,}개",
            )
        )
    else:
        criteria.append(_criterion("자동 제외 자료 비율", "%", None, None, "처리된 자료가 없습니다."))

    n_pass = sum(1 for c in criteria if c["status"] == "pass")
    n_fail = sum(1 for c in criteria if c["status"] == "fail")
    n_not_evaluated = sum(1 for c in criteria if c["status"] == "not_evaluated")

    return {
        "criteria": criteria,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_not_evaluated": n_not_evaluated,
        "overall_status": "fail" if n_fail > 0 else ("pass" if n_pass > 0 else "not_evaluated"),
    }
