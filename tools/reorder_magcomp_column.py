"""자력자료 CSV에서 MagComp 열을 MagLPF 바로 뒤로 옮겨,
Mag, MagLPF, MagComp 순서가 되도록 다시 저장하는 도구.

MagComp 열을 통째로 빼서 MagLPF 다음에 끼워 넣습니다. 그 사이에 있던
열들은 자연히 한 칸씩 밀립니다. 헤더 이름으로 찾으므로 파일마다 열 위치가
달라도 상관없습니다.

고친 파일은 원본을 건드리지 않고 "<원래이름>-r.csv" 로 따로 저장합니다.

이 도구가 지키는 원칙 - 열 순서 말고는 파일 내용을 바꾸지 않습니다:

  * 각 칸의 원본 문자열을 그대로 옮겨 붙입니다. 숫자를 다시 포맷하지
    않으므로 1.230 이 1.23 으로 바뀌는 일이 없습니다.
  * 줄바꿈 문자(CRLF/LF)와 파일 인코딩(BOM 유무 포함)을 원본 그대로
    유지합니다.
  * MagArrow 계열이 GGA/RMC 문장을 작은따옴표로 감싸 넣고 그 안에 쉼표가
    있으므로, 따옴표 안의 쉼표는 칸 구분으로 세지 않습니다.
  * 이 형식의 자료행은 줄 끝에 쉼표가 하나 더 붙어 헤더보다 칸이 하나
    많습니다. 그 꼬리 칸은 건드리지 않고 끝에 그대로 둡니다.

사용법 (Windows):

    python tools\\reorder_magcomp_column.py "D:\\HaeNam_Mag\\Magnetometer_old2_comp_mod"

또는 저장소 폴더의 MagComp_열순서.bat 을 더블클릭하세요.
바뀔 내용만 먼저 보려면 --dry-run 을 붙이세요.

표준 라이브러리만 사용하므로 venv/pandas 없이도 실행됩니다.
"""
from __future__ import annotations

import argparse
import codecs
import pathlib
import sys
from dataclasses import dataclass

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fix_magcomp_column import split_raw_fields  # noqa: E402  (같은 따옴표 규칙을 공유)

_DATA_SUFFIXES = {".csv", ".txt", ".dat"}
_OUTPUT_TAG = "-r"
_DELIM = ","
_ENCODINGS = ("utf-8", "cp949")


def _split_line_ending(line: str) -> tuple[str, str]:
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return line[: -len(ending)], ending
    return line, ""


def _find_column(header: list[str], name: str) -> int | None:
    """헤더에서 열 번호 찾기. 공백과 따옴표를 털어내고 정확히 비교한 뒤,
    못 찾으면 대소문자만 무시해 한 번 더 봅니다 - 장비/내보내기 설정에
    따라 MagLPF/maglpf 처럼 표기가 달라지는 경우가 있습니다."""
    cleaned = [c.strip().strip("'\"").strip() for c in header]
    if name in cleaned:
        return cleaned.index(name)
    lowered = [c.lower() for c in cleaned]
    if name.lower() in lowered:
        return lowered.index(name.lower())
    return None


def new_column_order(n_columns: int, move_index: int, after_index: int) -> list[int]:
    """move_index 열을 빼서 after_index 열 바로 뒤에 끼워 넣은 순서.

    빼고 나면 뒤쪽 열 번호가 하나씩 당겨지므로, 끼워 넣을 자리는 남은
    순서에서 after_index 를 다시 찾아 정합니다 - MagComp 가 MagLPF
    앞에 있든 뒤에 있든 똑같이 맞습니다."""
    order = list(range(n_columns))
    order.remove(move_index)
    return order[: order.index(after_index) + 1] + [move_index] + order[order.index(after_index) + 1 :]


@dataclass
class FileReport:
    path: pathlib.Path
    encoding: str = ""
    out_path: pathlib.Path | None = None
    n_columns: int = 0
    move_from: int | None = None
    move_to: int | None = None
    n_rows_reordered: int = 0
    n_rows_wrong_width: int = 0
    row_widths: dict | None = None
    skipped: str | None = None


def reorder_file(
    path: pathlib.Path,
    move_name: str,
    after_name: str,
    dry_run: bool,
    output_dir: pathlib.Path | None,
    suffix: str = _OUTPUT_TAG,
) -> FileReport:
    report = FileReport(path=path, row_widths={})

    raw = path.read_bytes()
    if raw.startswith(codecs.BOM_UTF8):
        text, report.encoding = raw.decode("utf-8-sig"), "utf-8-sig"
    else:
        for enc in _ENCODINGS:
            try:
                text, report.encoding = raw.decode(enc), enc
                break
            except UnicodeDecodeError:
                continue
        else:
            report.skipped = "인코딩을 알 수 없습니다 (utf-8/cp949 아님)."
            return report

    lines = text.splitlines(keepends=True)
    header_idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if header_idx is None:
        report.skipped = "빈 파일입니다."
        return report

    header_body, _ = _split_line_ending(lines[header_idx])
    header = split_raw_fields(header_body)
    report.n_columns = len(header)

    move_index = _find_column(header, move_name)
    after_index = _find_column(header, after_name)
    if move_index is None or after_index is None:
        missing = [n for n, i in ((move_name, move_index), (after_name, after_index)) if i is None]
        report.skipped = f"헤더에서 열을 찾지 못했습니다: {', '.join(missing)}"
        return report
    if move_index == after_index + 1:
        report.skipped = f"이미 {after_name} 바로 뒤에 {move_name} 이 있습니다 - 그대로 둡니다."
        report.move_from = report.move_to = move_index
        return report

    order = new_column_order(len(header), move_index, after_index)
    report.move_from = move_index
    report.move_to = order.index(move_index)

    out_lines: list[str] = []
    for i, line in enumerate(lines):
        body, ending = _split_line_ending(line)
        if not body.strip():
            out_lines.append(line)
            continue
        fields = split_raw_fields(body)
        report.row_widths[len(fields)] = report.row_widths.get(len(fields), 0) + 1
        if len(fields) < len(header):
            # 헤더보다 칸이 적은 줄은 어느 열이 어느 값인지 확정할 수 없어
            # 손대지 않습니다. 그대로 두면 그 줄만 옛 순서로 남으므로
            # 반드시 보고합니다.
            if i != header_idx:
                report.n_rows_wrong_width += 1
            out_lines.append(line)
            continue
        # 이 형식의 자료행은 줄 끝 쉼표 때문에 헤더보다 칸이 하나 많습니다.
        # 앞쪽 헤더 폭까지만 순서를 바꾸고 나머지 꼬리는 그대로 둡니다.
        head, tail = fields[: len(header)], fields[len(header) :]
        out_lines.append(_DELIM.join([head[k] for k in order] + tail) + ending)
        if i != header_idx:
            report.n_rows_reordered += 1

    target_dir = output_dir or path.parent
    report.out_path = target_dir / f"{path.stem}{suffix}{path.suffix}"
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(report.out_path, "w", encoding=report.encoding, newline="") as f:
            f.writelines(out_lines)
    return report


def _iter_input_files(folder: pathlib.Path, suffix: str) -> list[pathlib.Path]:
    return [
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in _DATA_SUFFIXES and not p.stem.endswith(suffix)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MagComp 열을 MagLPF 바로 뒤로 옮깁니다.")
    parser.add_argument("folder", nargs="?", default=r"D:\HaeNam_Mag\Magnetometer_old2_comp_mod",
                        help="자력자료 폴더 (하위 폴더까지 훑습니다)")
    parser.add_argument("--move", default="MagComp", help="옮길 열 이름 (기본 MagComp)")
    parser.add_argument("--after", default="MagLPF", help="이 열 바로 뒤로 옮깁니다 (기본 MagLPF)")
    parser.add_argument("--suffix", default=_OUTPUT_TAG, help=f"결과 파일 이름 꼬리표 (기본 {_OUTPUT_TAG})")
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 무엇이 바뀔지만 보여줍니다")
    parser.add_argument("--output-dir", default=None, help="결과를 다른 폴더에 모아 저장")
    args = parser.parse_args(argv)

    folder = pathlib.Path(args.folder)
    if not folder.is_dir():
        print(f"[오류] 폴더를 찾을 수 없습니다: {folder}")
        return 1

    files = _iter_input_files(folder, args.suffix)
    if not files:
        print(f"[알림] {folder} 에서 처리할 파일을 찾지 못했습니다 ({', '.join(sorted(_DATA_SUFFIXES))}).")
        return 1

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    print(f"대상 폴더 : {folder}")
    print(f"옮기기    : '{args.move}' 열 -> '{args.after}' 열 바로 뒤")
    print(f"모드      : {'미리보기 (파일을 쓰지 않음)' if args.dry_run else '실제 저장'}")
    print(f"파일 수   : {len(files)}")
    print("-" * 78)

    n_done = n_skipped = 0
    for p in files:
        r = reorder_file(p, args.move, args.after, args.dry_run, output_dir, args.suffix)
        if r.skipped:
            n_skipped += 1
            print(f"  [건너뜀] {p.name}: {r.skipped}")
            continue
        n_done += 1
        print(f"  {r.n_rows_reordered:>8,}행  {p.name}  "
              f"({args.move}: {r.move_from + 1}번째 열 -> {r.move_to + 1}번째, 전체 {r.n_columns}열)")
        if r.out_path:
            print(f"             -> {r.out_path.name}")
        if r.n_rows_wrong_width:
            print(f"             [주의] {r.n_rows_wrong_width:,}행은 칸 수가 헤더({r.n_columns})보다 적어 "
                  "순서를 바꾸지 못하고 그대로 두었습니다 - 그 줄만 옛 순서로 남습니다.")
        if r.row_widths and len(r.row_widths) > 2:
            shape = ", ".join(f"{k}칸 {v:,}행" for k, v in sorted(r.row_widths.items()))
            print(f"             [참고] 줄마다 칸 수가 여러 가지입니다: {shape}")

    print("-" * 78)
    print(f"합계: {n_done}개 파일 처리, {n_skipped}개 건너뜀"
          f"{' (미리보기 - 저장하지 않음)' if args.dry_run else ''}")
    if not args.dry_run and n_done:
        print(f"저장: 원본은 그대로 두고 '원래이름{args.suffix}.csv' 로 새로 만들었습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
