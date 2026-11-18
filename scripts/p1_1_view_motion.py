"""
p1_1_view_motion.py
-------------------
키포인트 파일 하나를 골라 Viser로 재생한다. (과제 2번 '문제 찾기' / 4번 '전후 비교' 캡처용)

사용법 (final_project 폴더에서):
    python scripts/p1_1_view_motion.py --list                       # 파일 목록
    python scripts/p1_1_view_motion.py hammer/s8LRGALu_h8           # 원본만 재생
    python scripts/p1_1_view_motion.py hammer/s8LRGALu_h8 --compare provided
        → 왼쪽: 원본 / 오른쪽: 교수님 제공 후처리(높이 보정)
    python scripts/p1_1_view_motion.py hammer/s8LRGALu_h8 --compare enhanced
        → 왼쪽: 교수님 제공 후처리 / 오른쪽: 개선 후처리(스무딩 등)

옵션:
    --start 10 --end 20   : 10~20초 구간만 재생
    --fps 15              : 느리게 재생 (기본은 영상 fps 30)

브라우저에서 http://localhost:8080 (터미널에 표시되는 주소) 를 열면 된다. 종료는 Ctrl+C.
"""

import os
import sys
import argparse
from glob import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from p1_2_post_processing_enhanced import (   # noqa: E402
    RAW_KEYPOINTS_DIR, load_raw_keypoints_full,
    create_height_corrected_target, refine, evaluate,
)


def resolve(name):
    p = name if name.endswith(".npz") else name + ".npz"
    for cand in (p, os.path.join(RAW_KEYPOINTS_DIR, p)):
        if os.path.exists(cand):
            return cand
    hits = glob(os.path.join(RAW_KEYPOINTS_DIR, "**", os.path.basename(p)), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"파일을 찾을 수 없음: {name}  (--list 로 목록 확인)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="예: hammer/s8LRGALu_h8")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--compare", choices=["provided", "enhanced"])
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--offset", type=float, default=1.5, help="두 스켈레톤 간격(m)")
    ap.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 끄기")
    args = ap.parse_args()

    if args.list or not args.name:
        for p in sorted(glob(os.path.join(RAW_KEYPOINTS_DIR, "**", "*.npz"), recursive=True)):
            seq, _, det = load_raw_keypoints_full(p)
            rel = os.path.relpath(p, RAW_KEYPOINTS_DIR).replace(os.sep, "/")[:-4]
            print(f"  {rel:<32} {len(det):>6} frames  ({len(det) / 30:5.1f}s)  검출 {det.mean() * 100:5.1f}%")
        return

    path = resolve(args.name)
    seq_full, vis, detected = load_raw_keypoints_full(path)
    raw = seq_full[detected]                       # 원본 (교수님 코드와 동일하게 검출 프레임만)
    provided = create_height_corrected_target(raw)

    if args.compare == "enhanced":
        enhanced, log = refine(seq_full, vis, fps=30.0)   # 처리 fps 는 데이터 기준(30), 재생 fps 와 별개
        left, right = provided, enhanced[detected]
        print("왼쪽: 제공 후처리(높이 보정)   |   오른쪽: 개선 후처리(애니메이터식: 키 리덕션 + 발 고정)")
        print(f"  missing frames {log['missing_frames']}, outlier joints removed {log['outlier_joints_removed']}")
    elif args.compare == "provided":
        left, right = raw, provided
        print("왼쪽: 원본(RAW)   |   오른쪽: 제공 후처리(높이 보정)")
    else:
        left, right = raw, None
        print("원본(RAW) 재생")

    # 구간 자르기
    s = int(args.start * 30) if args.start else 0
    e = int(args.end * 30) if args.end else None
    left = left[s:e]
    right = right[s:e] if right is not None else None

    print(f"\n{os.path.relpath(path, RAW_KEYPOINTS_DIR)}  — {len(left)} frames")
    print("  LEFT :", evaluate(left))
    if right is not None:
        print("  RIGHT:", evaluate(right))

    from utils.viser_test import PoseViser
    viewer = PoseViser(fps=int(args.fps))
    # 데이터와 바닥 그리드는 Y-up(유니티식) 기준인데 Viser 기본값은 Z-up → 위쪽 축을 +Y로 맞춤
    try:
        viewer.server.scene.set_up_direction("+y")
    except Exception as e:
        print("  (up direction 설정 실패:", e, ")")
    if right is not None:
        print("  색상: 파란 점+주황 선 = LEFT(원본쪽) / 보라 점+초록 선 = RIGHT(처리본, +X로 이동)")

    # 실제로 열린 포트로 브라우저 열기 (위의 'localhost:4242' 안내는 무시)
    try:
        port = viewer.server.get_port()
    except Exception:
        port = 8080
    url = f"http://localhost:{port}"
    print("\n" + "=" * 50)
    print(f"  ▶ 실제 주소: {url}   (4242 아님)")
    print("=" * 50 + "\n")
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)
    if right is None:
        viewer.play_sequence(left)
    else:
        viewer.play_two_sequences(left, right, offset=args.offset)


if __name__ == "__main__":
    main()
