"""
p1_5_make_work_clips.py
-----------------------
데모(Unity 시연)용 '작업 구간' 클립 만들기. 개선 후처리 결과(*_refined.npy)를 읽어서:

  1) 누운 프레임 보정: 상체(어깨 중심 - 골반 중심)가 수직에서 70° 넘게 누운 프레임은 BlazePose 추정 실패로 보고
     앞뒤 정상 프레임 사이를 선형 보간으로 메운다.
  2) 작업 구간 추출: 손목 최저점(임팩트 = 해머·삽이 땅에 닿는 순간)이 0.6~3.5초 간격으로 3번 이상 이어지는
     구간 중 가장 긴 것을 고른다 → 튜토리얼 영상의 설명 · 걷기 장면이 빠진다.
     시작과 끝을 둘 다 임팩트 프레임에서 자르므로 반복 재생할 때 이음새가 같은 자세(타격 순간)라 자연스럽다.
  3) data/processed/<clip>_work.csv (Unity 형식) + work_clips.json(요약)

사용법 (final_project 폴더에서):  python scripts/p1_5_make_work_clips.py
"""
import os
import sys
import json
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_2_post_processing_enhanced import OUT_DIR, export_unity_csv     # noqa: E402
from animator_cleanup import detect_impact_keys                         # noqa: E402
from rigid_tool import apply_rigid_tool, grip_metrics                   # noqa: E402

FPS = 30.0
TILT_MAX = 70.0
GAP_MIN, GAP_MAX = 0.6, 3.5
MIN_IMPACTS = 3


def torso_tilt(seq):
    hip = (seq[:, 23] + seq[:, 24]) / 2
    sh = (seq[:, 11] + seq[:, 12]) / 2
    up = sh - hip
    cos = up[:, 1] / np.linalg.norm(up, axis=1).clip(1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def repair_lying(seq):
    bad = torso_tilt(seq) > TILT_MAX
    if not bad.any():
        return seq.copy(), 0
    out = seq.copy()
    t = np.arange(len(seq))
    good = ~bad
    if good.sum() < 2:
        return out, int(bad.sum())
    flat = out.reshape(len(seq), -1)
    for c in range(flat.shape[1]):
        flat[bad, c] = np.interp(t[bad], t[good], flat[good, c])
    return flat.reshape(seq.shape), int(bad.sum())


def best_work_segment(impacts):
    """간격이 GAP_MIN~GAP_MAX 초인 임팩트가 이어지는 가장 긴 구간 (start_impact, end_impact)."""
    if len(impacts) < MIN_IMPACTS:
        return None
    best, run_start = None, 0
    for i in range(1, len(impacts) + 1):
        ok = i < len(impacts) and GAP_MIN * FPS <= impacts[i] - impacts[i - 1] <= GAP_MAX * FPS
        if not ok:
            if i - run_start >= MIN_IMPACTS:
                seg = (impacts[run_start], impacts[i - 1])
                if best is None or seg[1] - seg[0] > best[1] - best[0]:
                    best = seg
            run_start = i
    return best


def plot_grip(name, before, after, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(before)) / FPS
    gb = np.linalg.norm(before[:, 15] - before[:, 16], axis=1)
    ga = np.linalg.norm(after[:, 15] - after[:, 16], axis=1)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, gb * 100, color="#b0b0b0", lw=1, label="before (refined)")
    ax.plot(t, ga * 100, color="#d0661c", lw=1.6, label="after (rigid-tool constraint)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("hand gap (cm)")
    ax.set_title(f"{name}: distance between wrists on one handle")
    ax.legend(frameon=False, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    paths = sorted(glob(os.path.join(OUT_DIR, "*_refined.npy")))
    summary = {}
    for p in paths:
        name = os.path.basename(p)[:-len("_refined.npy")]
        seq = np.load(p)
        fixed, n_bad = repair_lying(seq)
        impacts, _ = detect_impact_keys(fixed, FPS)
        seg = best_work_segment(list(impacts))
        if seg is None:                      # 주기 동작을 못 찾으면 전체 사용
            a, b, how = 0, len(fixed) - 1, "whole clip (no periodic impacts)"
        else:
            a, b, how = int(seg[0]), int(seg[1]), "impact-to-impact"
        work = fixed[a:b + 1]
        n_imp = int(((impacts >= a) & (impacts <= b)).sum())
        tool = "hammer" if name.startswith("hammer") else "shovel"
        work_t, tinfo = apply_rigid_tool(work, FPS, tool)          # 강체 도구 제약 + 손 정리 + 도구 관절 33/34
        before, after = grip_metrics(work), grip_metrics(work_t[:, :33])
        export_unity_csv(work_t, os.path.join(OUT_DIR, f"{name}_work.csv"))
        np.save(os.path.join(OUT_DIR, f"{name}_work.npy"), work_t)
        plot_grip(name, work, work_t, os.path.join(OUT_DIR, f"{name}_grip.png"))
        summary[name] = {
            "frames_total": int(len(seq)), "lying_frames_repaired": n_bad,
            "work_start_s": round(a / FPS, 2), "work_end_s": round(b / FPS, 2),
            "work_len_s": round((b - a) / FPS, 2), "work_ratio(%)": round(100 * (b - a + 1) / len(seq), 1),
            "impacts_in_segment": n_imp, "segment": how,
            "rigid_tool": tinfo,
            "grip_before": {k: round(v, 4) for k, v in before.items()},
            "grip_after": {k: round(v, 4) for k, v in after.items()},
        }
        print(f"{name:28s} {summary[name]}")
    with open(os.path.join(OUT_DIR, "work_clips.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
