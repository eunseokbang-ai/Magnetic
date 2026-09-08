"""취득한 자력자료 파일을 취득 날짜별로 정리해 CSV 표로 만드는 도구.

폴더를 훑어 각 자력자료 파일을 이 앱의 로더(app.io_.drone_loader)로 그대로
읽습니다. 즉 날짜 판정 기준이 실제 처리 화면과 100% 동일합니다 - 파일명이
아니라 파일 안에 기록된 타임스탬프를 보므로, 파일명에 날짜가 없거나
파일명 날짜와 실제 측정 날짜가 다른 경우에도 맞게 나옵니다.

한 파일이 자정을 넘겨 두 날짜에 걸쳐 있는 경우를 위해 결과를 두 개로 냅니다:

  1. <출력>_파일별_요약.csv  - 파일 1개당 1행. '날짜수'와 '자정통과' 열로
     여러 날짜에 걸친 파일을 바로 골라낼 수 있고, '날짜목록'에 해당 날짜가
     모두 들어갑니다.
  2. <출력>_날짜별_상세.csv  - (파일 x 날짜) 조합당 1행. 두 날짜에 걸친
     파일은 2행으로 나뉘어 각 날짜에 몇 포인트가 몇 시부터 몇 시까지
     취득됐는지 보여줍니다. 날짜 기준으로 정렬/집계하기 좋은 형태입니다.

사용법 (Windows, run.bat을 한 번이라도 실행해 venv가 만들어진 상태):

    backend\\venv\\Scripts\\python.exe tools\\inventory_mag_files.py "C:\\magnetic\\HaeNam_Mag\\Magnetometer"

또는 같은 폴더의 자료목록_만들기.bat 을 더블클릭하세요.

시각(타임존) 주의: 이 도구는 파일에 기록된 시각을 그대로 씁니다. 장비가
UTC로 기록한다면 한국 시간과 9시간 차이가 나고, 그 때문에 실제로는 하루에
끝난 비행이 날짜를 넘긴 것처럼(또는 그 반대로) 보일 수 있습니다. 그런
경우 --tz-shift-hours 9 처럼 시간을 밀어서 한국 시간 기준으로 정리하세요.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.io_.drone_loader import DroneLoadError, load_drone_csv  # noqa: E402

# 자력자료로 시도해볼 확장자. 로더가 헤더를 보고 형식을 자동 판별하므로
# 확장자는 "읽어볼 후보"를 고르는 용도일 뿐입니다.
_DATA_SUFFIXES = {".csv", ".asc", ".txt", ".dat", ".xyz"}

_FORMAT_LABELS = {
    "generic": "일반/Geometrics MagArrow",
    "sensys_r1": "SENSYS MagDrone R1",
    "sensys_r3_raw": "SENSYS MagDrone R3 (raw)",
    "sensys_r3_asc": "SENSYS MagDrone R3 (ASC)",
    "microinfinity": "MicroInfinity Mag",
}


# 이 도구가 만들어내는 출력 파일 이름의 꼬리표. 출력을 대상 폴더 안에
# 저장하는 것이 기본이라, 두 번째 실행부터는 자기가 만든 CSV를 자료
# 파일로 착각해 '읽기 실패' 행으로 집어넣게 된다 - 이번 실행의 출력
# 경로뿐 아니라 이 꼬리표를 가진 이전 실행의 결과물도 함께 건너뛴다.
_OUTPUT_SUFFIXES = ("_파일별_요약.csv", "_날짜별_상세.csv")


def _candidate_files(root: pathlib.Path, recursive: bool, exclude: set[pathlib.Path]) -> list[pathlib.Path]:
    walker = root.rglob("*") if recursive else root.glob("*")
    files = [
        p
        for p in walker
        if p.is_file()
        and p.suffix.lower() in _DATA_SUFFIXES
        and p.resolve() not in exclude
        and not p.name.endswith(_OUTPUT_SUFFIXES)
    ]
    return sorted(files)


def _summarize(path: pathlib.Path, root: pathlib.Path, tz_shift_hours: float) -> tuple[dict, list[dict]]:
    """(파일별 요약 1행, 날짜별 상세 n행)을 만든다. 읽기 실패한 파일도
    조용히 빠지지 않고 '오류' 열이 채워진 요약 1행으로 남는다 - 목록에서
    통째로 사라지면 사용자가 누락을 알아채지 못하기 때문."""
    rel = str(path.relative_to(root))
    base = {
        "파일명": path.name,
        "상대경로": rel,
        "형식": "",
        "포인트수": 0,
        "시각없는_행": 0,
        "시작시각": "",
        "종료시각": "",
        "지속시간_분": "",
        "날짜수": 0,
        "날짜목록": "",
        "자정통과": "",
        "위도_최소": "",
        "위도_최대": "",
        "경도_최소": "",
        "경도_최대": "",
        "파일크기_MB": round(path.stat().st_size / 1024 / 1024, 2),
        "오류": "",
    }

    try:
        df = load_drone_csv(path)
    except (DroneLoadError, ValueError, KeyError, UnicodeDecodeError) as exc:
        base["오류"] = f"{type(exc).__name__}: {exc}"
        return base, []
    except Exception as exc:  # 예기치 못한 형식이라도 전체 실행이 멈추지 않게
        base["오류"] = f"읽기 실패({type(exc).__name__}): {exc}"
        return base, []

    # 포인트수는 '시각이 유효한 행' 기준. 날짜별 상세도 같은 행들로 만들기
    # 때문에, 이렇게 해야 상세 CSV의 날짜별 포인트수 합계가 요약 CSV의
    # 포인트수와 항상 일치한다. 시각을 못 읽은 행이 있으면 조용히 사라지지
    # 않도록 '시각없는_행'에 개수를 남긴다.
    ts_all = pd.to_datetime(df["timestamp"], errors="coerce")
    base["시각없는_행"] = int(ts_all.isna().sum())
    ts = ts_all.dropna()
    if tz_shift_hours:
        ts = ts + pd.Timedelta(hours=tz_shift_hours)
    if ts.empty:
        base["형식"] = _FORMAT_LABELS.get(df.attrs.get("source_format", ""), df.attrs.get("source_format", ""))
        base["오류"] = "유효한 타임스탬프가 없습니다."
        return base, []

    ts = ts.sort_values()
    dates = sorted(ts.dt.date.unique())
    fmt = df.attrs.get("source_format", "")

    base.update(
        {
            "형식": _FORMAT_LABELS.get(fmt, fmt),
            "포인트수": int(len(ts)),
            "시작시각": ts.iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "종료시각": ts.iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "지속시간_분": round((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60.0, 1),
            "날짜수": len(dates),
            "날짜목록": ", ".join(d.isoformat() for d in dates),
            "자정통과": "예" if len(dates) > 1 else "아니오",
            "위도_최소": round(float(df["lat"].min()), 6),
            "위도_최대": round(float(df["lat"].max()), 6),
            "경도_최소": round(float(df["lon"].min()), 6),
            "경도_최대": round(float(df["lon"].max()), 6),
        }
    )

    detail = []
    for i, day in enumerate(dates, start=1):
        same_day = ts[ts.dt.date == day]
        detail.append(
            {
                "취득날짜": day.isoformat(),
                "파일명": path.name,
                "상대경로": rel,
                "형식": base["형식"],
                "해당날짜_포인트수": int(len(same_day)),
                "해당날짜_시작시각": same_day.iloc[0].strftime("%H:%M:%S"),
                "해당날짜_종료시각": same_day.iloc[-1].strftime("%H:%M:%S"),
                "파일내_날짜순번": f"{i}/{len(dates)}",
                "자정통과파일": "예" if len(dates) > 1 else "아니오",
            }
        )
    return base, detail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="자력자료 파일을 취득 날짜별로 정리해 CSV로 저장합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("folder", help=r"자력자료 폴더 (예: C:\magnetic\HaeNam_Mag\Magnetometer)")
    parser.add_argument(
        "-o", "--out-prefix", default=None,
        help="출력 CSV 경로 접두사 (기본: 대상 폴더 안에 '자력자료목록')",
    )
    parser.add_argument(
        "--tz-shift-hours", type=float, default=0.0,
        help="기록된 시각에 더할 시간(시). 장비가 UTC로 기록하면 한국시간 기준 정리는 9 (기본: 0, 기록된 시각 그대로)",
    )
    parser.add_argument("--no-recursive", action="store_true", help="하위 폴더는 훑지 않음")
    args = parser.parse_args()

    root = pathlib.Path(args.folder)
    if not root.is_dir():
        print(f"[오류] 폴더를 찾을 수 없습니다: {root}")
        return 1

    # 출력 경로를 먼저 정해 스캔에서 제외한다 (위 _OUTPUT_SUFFIXES 설명 참고).
    prefix = pathlib.Path(args.out_prefix) if args.out_prefix else root / "자력자료목록"
    summary_path = prefix.with_name(prefix.name + "_파일별_요약.csv")
    detail_path = prefix.with_name(prefix.name + "_날짜별_상세.csv")

    files = _candidate_files(
        root,
        recursive=not args.no_recursive,
        exclude={summary_path.resolve(), detail_path.resolve()},
    )
    if not files:
        print(f"[오류] {root} 에서 자력자료 파일을 찾지 못했습니다 (대상 확장자: {', '.join(sorted(_DATA_SUFFIXES))})")
        return 1

    print(f"대상 폴더: {root}")
    print(f"파일 {len(files)}개를 읽는 중...\n")

    summaries, details = [], []
    for i, path in enumerate(files, start=1):
        summary, detail = _summarize(path, root, args.tz_shift_hours)
        summaries.append(summary)
        details.extend(detail)
        status = f"오류: {summary['오류']}" if summary["오류"] else (
            f"{summary['포인트수']:,}점  {summary['날짜목록']}" + ("  <== 두 날짜 이상!" if summary["날짜수"] > 1 else "")
        )
        print(f"  [{i}/{len(files)}] {summary['상대경로']}  -  {status}")

    summary_df = pd.DataFrame(summaries).sort_values(["시작시각", "파일명"], kind="stable")
    detail_df = pd.DataFrame(details)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["취득날짜", "해당날짜_시작시각", "파일명"], kind="stable")

    # utf-8-sig: Excel이 한글 열 이름을 깨뜨리지 않고 열도록.
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    spanning = summary_df[summary_df["날짜수"] > 1]
    failed = summary_df[summary_df["오류"] != ""]
    all_dates = sorted({d["취득날짜"] for d in details})

    print("\n" + "=" * 60)
    print(f"파일 {len(summary_df)}개 중 정상 {len(summary_df) - len(failed)}개, 읽기 실패 {len(failed)}개")
    print(f"취득 날짜 {len(all_dates)}일: {', '.join(all_dates) if all_dates else '-'}")
    if len(spanning):
        print(f"\n두 날짜 이상에 걸친 파일 {len(spanning)}개:")
        for _, row in spanning.iterrows():
            print(f"  - {row['상대경로']}  ({row['날짜목록']})  {row['시작시각']} ~ {row['종료시각']}")
    else:
        print("\n두 날짜에 걸친 파일: 없음")
    if len(failed):
        print(f"\n읽지 못한 파일 {len(failed)}개 (CSV의 '오류' 열 참고):")
        for _, row in failed.iterrows():
            print(f"  - {row['상대경로']}: {row['오류']}")
    print("\n저장 완료:")
    print(f"  {summary_path}")
    print(f"  {detail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
