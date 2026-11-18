"""
p1_4_motion_studio.py
---------------------
Viser 인터랙티브 모션 비교 뷰어 (보너스: 시각화 개선)
Author: Junha Yoo (20242134)

교수님 제공 utils/viser_test.py 는 그대로 두고, Viser 의 GUI 패널 기능으로 확장한 별도 뷰어.
콘솔을 껐다 켤 필요 없이 브라우저 안에서:
  - 클립(영상) 전환
  - 재생 / 일시정지 / 프레임 슬라이더 / 재생 속도
  - 원본(RAW) · 제공 후처리 · 개선 후처리 레이어 켜고 끄기
  - 발목 궤적(trail) 표시 → 발 미끄러짐을 눈으로 확인
  - 클립별 평가 지표 표시

사용법 (final_project 폴더에서):
    python scripts/p1_4_motion_studio.py
브라우저가 자동으로 열린다. 종료는 콘솔에서 Ctrl+C.
"""

import os
import sys
import time
import webbrowser
from glob import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import viser  # noqa: E402
from p1_2_post_processing_enhanced import (  # noqa: E402
    RAW_KEYPOINTS_DIR, POSE_CONNECTIONS, load_raw_keypoints_full,
    create_height_corrected_target, refine, evaluate,
)

FPS = 30.0
LAYERS = {
    # key: (표시 이름, 점 색, 선 색)
    "raw":      ("RAW (원본)",              (0.60, 0.62, 0.65), (0.45, 0.47, 0.50)),
    "provided": ("Provided (제공 후처리)",   (0.95, 0.45, 0.40), (0.85, 0.33, 0.31)),
    "refined":  ("Refined (개선 후처리)",    (0.30, 0.60, 0.95), (0.17, 0.48, 0.73)),
}
ANKLES = (27, 28)


# ---------------------------------------------------------------- data
def list_clips():
    paths = sorted(glob(os.path.join(RAW_KEYPOINTS_DIR, "**", "*.npz"), recursive=True))
    return {os.path.relpath(p, RAW_KEYPOINTS_DIR).replace(os.sep, "/")[:-4]: p for p in paths}


_cache = {}


def load_clip(path):
    if path in _cache:
        return _cache[path]
    seq_full, vis, det = load_raw_keypoints_full(path)
    raw = seq_full[det]
    provided = create_height_corrected_target(raw)
    refined, log = refine(seq_full, vis, fps=FPS)
    refined = refined[det]
    data = {"raw": raw, "provided": provided, "refined": refined,
            "metrics": {k: evaluate(v, FPS) for k, v in (("raw", raw), ("provided", provided), ("refined", refined))},
            "log": {k: v for k, v in log.items() if not k.startswith("_")}}
    _cache[path] = data
    return data


def metrics_markdown(d):
    keys = [("foot_slide(m/s)", "발 미끄러짐 (m/s)"),
            ("jitter_acc(m/s^2)", "떨림 (m/s²)"),
            ("bone_length_CV", "뼈 길이 변동"),
            ("pelvis_vert_jitter(m/frame)", "몸 상하 떨림 (m/f)"),
            ("root_travel(m)", "루트 이동 (m)")]
    rows = ["| 지표 | RAW | Provided | Refined |", "|---|---|---|---|"]
    for k, lab in keys:
        rows.append(f"| {lab} | {d['metrics']['raw'][k]} | {d['metrics']['provided'][k]} | **{d['metrics']['refined'][k]}** |")
    lg = d["log"]
    extra = []
    if "keys" in lg:
        extra.append(f"키 {lg['keys']}개 ({lg['key_ratio(%)']}%) · 임팩트 {lg['impact_keys']}회")
    if "swing_period(s)" in lg:
        extra.append(f"휘두르기 주기 {lg['swing_period(s)']}s")
    return "\n".join(rows) + ("\n\n" + " · ".join(extra) if extra else "")


# ---------------------------------------------------------------- viewer
def main():
    clips = list_clips()
    if not clips:
        print("키포인트 파일이 없습니다:", RAW_KEYPOINTS_DIR)
        return
    names = list(clips.keys())

    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/ground", width=6.0, height=6.0, plane="xz",
                          cell_size=0.2, section_size=1.0, position=(0.0, 0.0, 0.0))

    # --- GUI
    with server.gui.add_folder("Clip"):
        gui_clip = server.gui.add_dropdown("영상", tuple(names), initial_value=names[0])
        gui_info = server.gui.add_markdown("불러오는 중...")
    with server.gui.add_folder("Playback"):
        gui_play = server.gui.add_checkbox("재생", initial_value=True)
        gui_frame = server.gui.add_slider("프레임", min=0, max=1, step=1, initial_value=0)
        gui_speed = server.gui.add_slider("속도", min=0.1, max=2.0, step=0.1, initial_value=1.0)
        gui_time = server.gui.add_markdown("")
    with server.gui.add_folder("Layers"):
        gui_layer = {k: server.gui.add_checkbox(LAYERS[k][0], initial_value=True) for k in LAYERS}
        gui_gap = server.gui.add_slider("간격 (m)", min=0.5, max=3.0, step=0.1, initial_value=1.5)
        gui_trail = server.gui.add_checkbox("발목 궤적 표시", initial_value=True)
        gui_trail_len = server.gui.add_slider("궤적 길이 (초)", min=0.5, max=5.0, step=0.5, initial_value=2.0)
    with server.gui.add_folder("Metrics"):
        gui_metrics = server.gui.add_markdown("")

    # --- scene handles
    n_seg = len(POSE_CONNECTIONS)
    handles = {}
    for k, (label, pc, lc) in LAYERS.items():
        handles[k] = {
            "pts": server.scene.add_point_cloud(f"/{k}/points", points=np.zeros((33, 3)),
                                                colors=np.tile(np.array([pc]), (33, 1)), point_size=0.03),
            "lines": server.scene.add_line_segments(f"/{k}/lines", points=np.zeros((n_seg, 2, 3)),
                                                    colors=np.tile(np.array([lc]), (n_seg, 2, 1)), line_width=3.0),
            "trail": server.scene.add_line_segments(f"/{k}/trail", points=np.zeros((1, 2, 3)),
                                                    colors=np.tile(np.array([pc]), (1, 2, 1)), line_width=2.0),
            "label": server.scene.add_label(f"/{k}/label", text=label, position=(0.0, 2.0, 0.0)),
        }

    state = {"data": None, "frame": 0.0, "T": 1, "syncing": False}

    def set_clip(name):
        gui_info.content = f"**{name}** 불러오는 중..."
        d = load_clip(clips[name])
        state["data"] = d
        state["T"] = len(d["raw"])
        state["frame"] = 0.0
        state["syncing"] = True
        gui_frame.max = state["T"] - 1
        gui_frame.value = 0
        state["syncing"] = False
        gui_info.content = f"**{name}** — {state['T']} frames ({state['T'] / FPS:.1f}s)"
        gui_metrics.content = metrics_markdown(d)

    @gui_clip.on_update
    def _(_):
        set_clip(gui_clip.value)

    @gui_frame.on_update
    def _(_):
        if not state["syncing"]:
            state["frame"] = float(gui_frame.value)

    def draw(t):
        d = state["data"]
        order = [k for k in LAYERS if gui_layer[k].value]
        gap = gui_gap.value
        # 켜진 레이어만 가운데 정렬해서 배치
        offsets = {k: (i - (len(order) - 1) / 2) * gap for i, k in enumerate(order)}
        trail_n = int(gui_trail_len.value * FPS)
        for k in LAYERS:
            h = handles[k]
            visible = k in offsets
            for part in ("pts", "lines", "label", "trail"):
                h[part].visible = visible
            if not visible:
                continue
            seq = d[k]
            off = np.array([offsets[k], 0.0, 0.0])
            p = seq[t] + off
            h["pts"].points = p
            h["lines"].points = np.array([[p[a], p[b]] for a, b in POSE_CONNECTIONS])
            h["label"].position = tuple(p[0] + np.array([0.0, 0.35, 0.0]))
            if gui_trail.value and t > 1:
                s0 = max(0, t - trail_n)
                segs = []
                for j in ANKLES:
                    path = seq[s0:t + 1, j] + off
                    segs.append(np.stack([path[:-1], path[1:]], axis=1))
                segs = np.concatenate(segs, axis=0)
                h["trail"].points = segs
                h["trail"].colors = np.tile(np.array([LAYERS[k][1]]), (len(segs), 2, 1))
                h["trail"].visible = True
            else:
                h["trail"].visible = False

    set_clip(names[0])

    port = server.get_port()
    url = f"http://localhost:{port}"
    print("\n" + "=" * 50)
    print(f"  ▶ Motion Studio: {url}")
    print("  종료: Ctrl+C")
    print("=" * 50 + "\n")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)

    try:
        last = time.time()
        while True:
            now = time.time()
            dt = now - last
            last = now
            if state["data"] is not None:
                if gui_play.value:
                    state["frame"] = (state["frame"] + dt * FPS * gui_speed.value) % state["T"]
                    state["syncing"] = True
                    gui_frame.value = int(state["frame"])
                    state["syncing"] = False
                t = int(state["frame"]) % state["T"]
                draw(t)
                gui_time.content = f"{t / FPS:6.2f}s / {state['T'] / FPS:.2f}s"
            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        print("종료")


if __name__ == "__main__":
    main()
