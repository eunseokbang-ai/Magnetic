"""자력자료 CSV에서 MagComp 값이 제 열(기본 BK)이 아니라 오른쪽 열(기본 CK)로
밀려 들어간 행을 찾아 제자리로 옮겨 다시 저장하는 도구.

고친 파일은 원본을 건드리지 않고 "<원래이름>-m.csv" 로 따로 저장합니다.

이 도구가 지키는 원칙 - 옮기는 두 칸 말고는 파일을 한 글자도 바꾸지 않습니다:

  * 고칠 것이 없는 줄은 원본 줄을 그대로 씁니다. 다시 만들지 않습니다.
  * 고치는 줄도 해당 두 칸만 바꾸고 나머지 칸은 원본 문자열 그대로 이어
    붙입니다. 숫자를 다시 포맷하지 않으므로 자릿수/소수점 표기가 변하지
    않습니다. (pandas로 읽었다 쓰면 1.230 -> 1.23 처럼 값이 미묘하게
    바뀌는데, 원자료를 다루는 도구가 해서는 안 되는 일입니다.)
  * 줄바꿈 문자(CRLF/LF)와 파일 인코딩을 원본 그대로 유지합니다.

MagArrow 계열 CSV는 GGA/RMC 문장을 작은따옴표(')로 감싸 넣는데 그 안에
쉼표가 들어 있습니다. 그래서 쉼표로 그냥 자르면 열이 어긋납니다 - 이
도구는 따옴표 안의 쉼표를 무시하고 자릅니다. (덧붙여, 이 따옴표가 일부
행에서 빠지면 뒤쪽 열이 통째로 밀리는데, 이번 증상의 원인일 수 있어
보고서에 따로 표시합니다.)

사용법 (Windows):

    python tools\\fix_magcomp_column.py "C:\\magnetic\\HaeNam_Mag\\Magnetometer"

또는 저장소 폴더의 MagComp_열정리.bat 을 더블클릭하세요.
먼저 무엇이 바뀔지만 보려면 --dry-run 을 붙이세요.

표준 라이브러리만 사용하므로 venv/pandas 없이도 실행됩니다.
"""
from __future__ import annotations

import argparse
import codecs
import pathlib
import sys
from dataclasses import dataclass, field

# 대상 확장자. 자력자료 원본만 고르기 위한 후보 목록입니다.
_DATA_SUFFIXES = {".csv", ".txt", ".dat"}

# 이 도구가 붙이는 꼬리표. 두 번째 실행 때 자기가 만든 결과물을 다시
# 입력으로 집어 "-m-m.csv" 를 만드는 것을 막습니다.
_OUTPUT_TAG = "-m"

# MagArrow 계열 CSV가 GGA/RMC 문장을 감쌀 때 쓰는 따옴표.
_QUOTE = "'"
_DELIM = ","

# 읽기를 시도할 인코딩 순서. 쓸 때도 읽은 인코딩을 그대로 씁니다.
# BOM 유무는 따로 봅니다: BOM 없는 UTF-8 파일을 utf-8-sig 로 읽으면
# 멀쩡히 읽히지만, 그 이름으로 다시 쓰면 파이썬이 BOM을 새로 붙여
# 원본에 없던 3바이트가 파일 앞에 생깁니다. BOM에 걸려 넘어지는
# 프로그램이 아직 흔하므로, 원본에 있던 그대로만 씁니다.
_ENCODINGS = ("utf-8", "cp949")


def column_letter_to_index(letter: str) -> int:
    """엑셀 열 이름을 0부터 시작하는 번호로 (A->0, BK->62, CK->88)."""
    letter = letter.strip().upper()
    if not letter or not letter.isalpha():
        raise ValueError(f"열 이름이 올바르지 않습니다: {letter!r} (A, BK, CK 처럼 적으세요)")
    index = 0
    for ch in letter:
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return index - 1


def split_raw_fields(line: str, delim: str = _DELIM, quote: str = _QUOTE) -> list[str]:
    """따옴표 안의 구분자를 무시하고 한 줄을 칸으로 자릅니다. 잘라낸 조각은
    따옴표까지 포함한 원본 문자열 그대로라, delim으로 다시 이어 붙이면
    원래 줄이 글자 하나까지 똑같이 복원됩니다 - 이 도구가 손대지 않은 칸을
    절대 건드리지 않는다고 말할 수 있는 근거입니다."""
    fields: list[str] = []
    start = 0
    in_quote = False
    for i, ch in enumerate(line):
        if ch == quote:
            in_quote = not in_quote
        elif ch == delim and not in_quote:
            fields.append(line[start:i])
            start = i + 1
    fields.append(line[start:])
    return fields


def _split_line_ending(line: str) -> tuple[str, str]:
    """줄 본문과 줄바꿈 문자를 분리 (CRLF/LF/없음 그대로 보존)."""
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return line[: -len(ending)], ending
    return line, ""


@dataclass
class FileReport:
    path: pathlib.Path
    encoding: str = ""
    out_path: pathlib.Path | None = None
    n_lines: int = 0
    n_moved: int = 0          # CK -> BK 로 옮긴 행
    n_already_ok: int = 0     # BK에 이미 값이 있던 행
    n_both_filled: int = 0    # BK와 CK 둘 다 값이 있어 손대지 않은 행
    n_neither: int = 0        # 둘 다 비어 있던 행
    n_too_short: int = 0      # 칸 수가 모자라 판단할 수 없던 행
    field_counts: dict = field(default_factory=dict)   # 칸 수 -> 행 수
    unbalanced_quote_lines: int = 0
    error: str | None = None

    @property
    def changed(self) -> bool:
        return self.n_moved > 0


def fix_file(
    path: pathlib.Path,
    bk_index: int,
    ck_index: int,
    dry_run: bool,
    write_unchanged: bool,
    output_dir: pathlib.Path | None,
) -> FileReport:
    report = FileReport(path=path)

    raw = path.read_bytes()
    if raw.startswith(codecs.BOM_UTF8):
        text = raw.decode("utf-8-sig")
        report.encoding = "utf-8-sig"   # 쓸 때 BOM을 다시 붙여 원본과 맞춥니다
    else:
        for encoding in _ENCODINGS:
            try:
                text = raw.decode(encoding)
                report.encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        else:
            report.error = "인코딩을 알 수 없어 건너뜁니다 (utf-8/cp949 아님)."
            return report

    # keepends=True: 줄바꿈을 그대로 들고 있다가 그대로 돌려줍니다.
    lines = text.splitlines(keepends=True)
    if not lines:
        report.error = "빈 파일입니다."
        return report

    out_lines: list[str] = []
    for line in lines:
        body, ending = _split_line_ending(line)
        if not body.strip():
            out_lines.append(line)
            continue

        report.n_lines += 1
        if body.count(_QUOTE) % 2 == 1:
            # 따옴표가 홀수 개 = 어딘가에서 짝이 맞지 않는다는 뜻.
            # 이런 줄은 칸 자르기 자체를 믿을 수 없으므로 손대지 않습니다.
            report.unbalanced_quote_lines += 1
            out_lines.append(line)
            continue

        fields = split_raw_fields(body)
        report.field_counts[len(fields)] = report.field_counts.get(len(fields), 0) + 1

        if len(fields) <= bk_index:
            report.n_too_short += 1
            out_lines.append(line)
            continue

        bk_filled = fields[bk_index].strip() != ""
        ck_filled = len(fields) > ck_index and fields[ck_index].strip() != ""

        if bk_filled and ck_filled:
            report.n_both_filled += 1
            out_lines.append(line)
        elif bk_filled:
            report.n_already_ok += 1
            out_lines.append(line)
        elif ck_filled:
            # 이 도구가 하는 유일한 수정: CK 칸의 원본 문자열을 그대로
            # BK 칸으로 옮기고 CK는 빈 칸으로 둡니다. 칸 수는 그대로라
            # 뒤쪽 열이 밀리지 않습니다.
            fields[bk_index] = fields[ck_index]
            fields[ck_index] = ""
            report.n_moved += 1
            out_lines.append(_DELIM.join(fields) + ending)
        else:
            report.n_neither += 1
            out_lines.append(line)

    if not report.changed and not write_unchanged:
        return report

    target_dir = output_dir or path.parent
    report.out_path = target_dir / f"{path.stem}{_OUTPUT_TAG}{path.suffix}"
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        # newline="" : 위에서 보존한 줄바꿈 문자를 파이썬이 다시 손대지 못하게.
        with open(report.out_path, "w", encoding=report.encoding, newline="") as f:
            f.writelines(out_lines)
    return report


def _iter_input_files(folder: pathlib.Path) -> list[pathlib.Path]:
    files = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _DATA_SUFFIXES:
            continue
        if p.stem.endswith(_OUTPUT_TAG):  # 이전 실행의 결과물
            continue
        files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="자력자료 CSV에서 오른쪽 열로 밀려 들어간 MagComp 값을 제자리로 옮깁니다.",
    )
    parser.add_argument("folder", nargs="?", default=r"C:\magnetic\HaeNam_Mag\Magnetometer",
                        help="자력자료 폴더 (하위 폴더까지 훑습니다)")
    parser.add_argument("--from-column", default="CK", help="값이 잘못 들어가 있는 열 (기본 CK)")
    parser.add_argument("--to-column", default="BK", help="값이 있어야 할 열 (기본 BK)")
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 무엇이 바뀔지만 보여줍니다")
    parser.add_argument("--write-unchanged", action="store_true",
                        help="고칠 것이 없는 파일도 -m 이름으로 복사본을 만듭니다")
    parser.add_argument("--output-dir", default=None, help="결과를 다른 폴더에 모아 저장 (기본: 원본과 같은 폴더)")
    args = parser.parse_args(argv)

    folder = pathlib.Path(args.folder)
    if not folder.is_dir():
        print(f"[오류] 폴더를 찾을 수 없습니다: {folder}")
        return 1

    try:
        bk_index = column_letter_to_index(args.to_column)
        ck_index = column_letter_to_index(args.from_column)
    except ValueError as exc:
        print(f"[오류] {exc}")
        return 1

    files = _iter_input_files(folder)
    if not files:
        print(f"[알림] {folder} 에서 처리할 파일을 찾지 못했습니다 ({', '.join(sorted(_DATA_SUFFIXES))}).")
        return 1

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    print(f"대상 폴더 : {folder}")
    print(f"옮기기    : {args.from_column}열({ck_index + 1}번째) -> {args.to_column}열({bk_index + 1}번째)")
    print(f"모드      : {'미리보기 (파일을 쓰지 않음)' if args.dry_run else '실제 저장'}")
    print(f"파일 수   : {len(files)}")
    print("-" * 78)

    reports = [fix_file(p, bk_index, ck_index, args.dry_run, args.write_unchanged, output_dir) for p in files]

    total_moved = 0
    n_changed_files = 0
    for r in reports:
        if r.error:
            print(f"  [건너뜀] {r.path.name}: {r.error}")
            continue
        total_moved += r.n_moved
        if r.changed:
            n_changed_files += 1
        status = f"{r.n_moved:>7,}행 이동" if r.changed else "      이동 없음"
        print(f"  {status}  {r.path.name}  (자료 {r.n_lines:,}행, 정상 {r.n_already_ok:,})")
        if r.out_path:
            print(f"           -> {r.out_path.name}")

        # 손대지 않았지만 알고 있어야 할 것들.
        if r.n_both_filled:
            print(f"           [확인] {r.n_both_filled:,}행은 {args.to_column}열과 {args.from_column}열에 "
                  f"값이 둘 다 있어 손대지 않았습니다.")
        if r.unbalanced_quote_lines:
            print(f"           [확인] {r.unbalanced_quote_lines:,}행은 작은따옴표 짝이 맞지 않아 "
                  "칸을 믿을 수 없어 손대지 않았습니다.")
        if r.n_too_short:
            print(f"           [확인] {r.n_too_short:,}행은 칸 수가 {bk_index + 1}개에 못 미쳐 판단할 수 없었습니다.")
        if len(r.field_counts) > 1:
            # 칸 수가 여러 가지라는 것은 행마다 열 구조가 다르다는 뜻입니다.
            # MagComp 한 칸만 옮겨서 될 일인지 판단할 근거가 되므로 꼭 보여줍니다.
            shape = ", ".join(f"{n}칸 {c:,}행" for n, c in sorted(r.field_counts.items()))
            print(f"           [주의] 행마다 칸 수가 다릅니다: {shape}")
            widths = sorted(r.field_counts)
            print(f"                  칸 수 차이 {widths[-1] - widths[0]}개 - 이 값이 "
                  f"{ck_index - bk_index}(= {args.to_column}->{args.from_column} 거리)와 같다면 "
                  "MagComp 한 칸이 아니라 그 뒤 열 전체가 밀린 것이므로 알려주세요.")

    print("-" * 78)
    print(f"합계: {n_changed_files}개 파일에서 {total_moved:,}행 이동"
          f"{' (미리보기 - 저장하지 않음)' if args.dry_run else ''}")
    if not args.dry_run and n_changed_files:
        print(f"저장: 원본은 그대로 두고 '원래이름{_OUTPUT_TAG}.csv' 형태로 새로 만들었습니다 (원본은 그대로입니다).")
        print(f"      고칠 것이 없던 파일은 -m 을 만들지 않았습니다. 전부 -m 이름으로 갖추려면 "
              "--write-unchanged 를 붙여 다시 실행하세요.")
    if total_moved == 0:
        print(f"밀려 있는 값을 찾지 못했습니다. {args.to_column}/{args.from_column} 열 지정이 맞는지 확인하세요 "
              f"(--to-column / --from-column 으로 바꿀 수 있습니다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
