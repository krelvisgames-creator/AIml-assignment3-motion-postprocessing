"""
p0_download_videos.py
---------------------
레퍼런스 유튜브 영상을 클래스별로 내려받고, 바로 BlazePose 3D 키포인트까지 추출한다.

  다운로드  →  인머러/Videos/<class>/<video_id>.mp4
  키포인트  →  final_project/data/raw_keypoints/<class>/<video_id>.npz   (p2_create_dataset.py 구조)

사용법 (final_project 폴더에서, ai_ml_final 환경 활성화 후):
    python scripts/p0_download_videos.py              # 다운로드 + 추출
    python scripts/p0_download_videos.py --no-extract # 다운로드만
    python scripts/p0_download_videos.py --only hammer

이미 받은 영상/추출한 파일은 건너뛴다. 실패한 영상은 마지막에 목록으로 출력.

다운로드는 yt-dlp 사용 (pytubefix 는 유튜브 봇 감지에 자주 막힘):
    pip install -U yt-dlp
그래도 막히면 브라우저 쿠키 사용:  --cookies-from chrome   (또는 edge, firefox)
그래도 안 되면 영상을 직접 받아 Videos/<class>/ 에 mp4 로 넣고 --extract-only 로 실행.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent                 # final_project
INMERE_DIR = ROOT.parents[2]                                     # ...\인머러
VIDEO_DIR = INMERE_DIR / "Videos"
KEYPOINT_DIR = ROOT / "data" / "raw_keypoints"

# ------------------------------------------------------------------
# 클래스별 레퍼런스 영상
#   (url, start_sec, end_sec)  — 구간 자르기가 필요 없으면 None
# ------------------------------------------------------------------
SOURCES = {
    "hammer": [
        ("https://m.youtube.com/shorts/s8LRGALu_h8", None, None),
        ("https://www.youtube.com/shorts/PAQI7y61TY0", None, None),
        ("https://www.youtube.com/watch?v=wFXKSWzuHe0", 23, 262),   # 0:23~4:22 시범 구간
        ("https://www.youtube.com/watch?v=851PGyIQEss", None, None),
    ],
    "shovel": [
        ("https://www.youtube.com/watch?v=g4mKvTxESFY", None, None),
        ("https://www.youtube.com/watch?v=HmyvWOXlSt0", None, None),
        ("https://www.youtube.com/watch?v=htjqQO1XASQ", None, None),
    ],
    # 3클래스로 갈 경우 주석 해제 + p3_train_mlp.py 의 ActionMLP(num_classes=3) 수정
    # "lift_box": [
    #     ("https://www.youtube.com/watch?v=O31u_T5wzJQ", None, None),
    #     ("https://www.youtube.com/watch?v=kIa5sa2C4oE", None, None),
    #     ("https://www.youtube.com/watch?v=IiGiPO_R_rU", None, None),
    # ],
}


def video_id(url):
    """URL → 유튜브 영상 ID"""
    import re
    m = re.search(r"(?:v=|shorts/|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url


def download(url, out_dir, cookies_from=None):
    import yt_dlp

    vid = video_id(url)
    out_path = out_dir / f"{vid}.mp4"
    if out_path.exists():
        print(f"  [skip] 이미 있음: {out_path.name}")
        return out_path

    opts = {
        # 소리는 필요 없으므로 mp4 영상 스트림만 (병합 불필요), 최대 1080p
        "format": "bv*[ext=mp4][height<=1080]/b[ext=mp4]/bv*/b",
        "outtmpl": str(out_dir / f"{vid}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if cookies_from:
        opts["cookiesfrombrowser"] = (cookies_from,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # mp4 가 아닌 확장자로 받아졌으면 이름만 찾아서 반환
    if not out_path.exists():
        cands = list(out_dir.glob(f"{vid}.*"))
        if not cands:
            raise RuntimeError("다운로드 파일을 찾을 수 없음")
        out_path = cands[0]
    print(f"  [down] {info.get('height')}p  {info.get('title')}")
    return out_path


def extract(video_path, out_path, start=None, end=None):
    import cv2
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=2, smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f0 = int(start * fps) if start else 0
    f1 = int(end * fps) if end else None
    if f0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = (min(f1, total_src) if f1 else total_src) - f0
    print(f"  [....] {video_path.name} 추출 시작 ({total} frames)", flush=True)

    rows, frame_idx, src_idx, detected = [], 0, f0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (f1 is not None and src_idx >= f1):
            break
        # 1080p 등 큰 영상은 720p 로 줄여서 처리 (속도↑, 월드 좌표 결과는 동일 스케일)
        h, w = frame.shape[:2]
        if h > 720:
            frame = cv2.resize(frame, (int(w * 720 / h), 720), interpolation=cv2.INTER_AREA)
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if frame_idx % 150 == 0 and frame_idx:
            print(f"         {frame_idx}/{total} ({frame_idx / max(total, 1) * 100:.0f}%)", flush=True)
        if res.pose_world_landmarks:
            detected += 1
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                rows.append({"frame": frame_idx, "landmark": i,
                             "x": lm.x, "y": -lm.y, "z": lm.z,
                             "visibility": lm.visibility})
        frame_idx += 1
        src_idx += 1
    cap.release()
    pose.close()

    np.savez(out_path, data=pd.DataFrame(rows).to_numpy())
    rate = detected / max(frame_idx, 1) * 100
    print(f"  [kpts] {frame_idx} frames, 검출률 {rate:.0f}%  (fps {fps:.0f}) → {out_path.name}")
    if rate < 80:
        print("         ⚠ 검출률 낮음 — 전신이 잘 안 나오는 영상일 수 있음")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true", help="다운로드만")
    ap.add_argument("--extract-only", action="store_true",
                    help="다운로드 없이 Videos/<class>/ 안의 영상 전부 추출 (직접 받은 영상 포함)")
    ap.add_argument("--only", help="특정 클래스만 (예: hammer)")
    ap.add_argument("--cookies-from", help="봇 감지 시 브라우저 쿠키 사용: chrome / edge / firefox")
    args = ap.parse_args()

    failed = []
    for cls, items in SOURCES.items():
        if args.only and cls != args.only:
            continue
        vdir = VIDEO_DIR / cls
        kdir = KEYPOINT_DIR / cls
        vdir.mkdir(parents=True, exist_ok=True)
        kdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {cls} ===")

        # 1) 다운로드
        trims = {video_id(u): (s0, s1) for u, s0, s1 in items}
        if not args.extract_only:
            for url, _, _ in items:
                try:
                    download(url, vdir, args.cookies_from)
                except Exception as e:
                    print(f"  [FAIL] {url}\n         {type(e).__name__}: {str(e)[:200]}")
                    failed.append(url)

        if args.no_extract:
            continue

        # 2) 폴더 안의 모든 영상 추출 (직접 넣은 영상도 포함)
        videos = sorted(p for p in vdir.iterdir()
                        if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov"))
        for vpath in videos:
            kpath = kdir / f"{vpath.stem}.npz"
            if kpath.exists():
                print(f"  [skip] 키포인트 있음: {kpath.name}")
                continue
            start, end = trims.get(vpath.stem, (None, None))
            try:
                extract(vpath, kpath, start, end)
            except Exception as e:
                print(f"  [FAIL] 추출 실패 {vpath.name}: {type(e).__name__}: {e}")

    print("\n완료.")
    print(f"영상:     {VIDEO_DIR}")
    print(f"키포인트: {KEYPOINT_DIR}")
    if failed:
        print("\n다운로드 실패한 영상 (직접 받아서 Videos/<class>/ 에 넣고 --extract-only 로 다시 실행):")
        for u in failed:
            print("  -", u)


if __name__ == "__main__":
    main()
