"""
analyze_failure_case.py
-----------------------
한계 분석: '제자리 동작' 가정이 깨지는 구간(사람이 걸어서 이동)에서 발 고정이 얼마나 덜 작동하는지 확인.
발 6점 중심의 3초 rolling median(= 실제 발 위치 추정)이 움직이는 속도로 '이동 구간'을 판정하고,
구간별 발 미끄러짐 감소율을 비교한다.

사용법: python scripts/analyze_failure_case.py shovel/HmyvWOXlSt0
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import p1_2_post_processing_enhanced as E          # noqa: E402
from animator_cleanup import FEET, _rolling_median  # noqa: E402

FPS = 30.0
MOVE_THRESH = 0.08     # m/s : 발 위치 추정치가 이보다 빨리 움직이면 '이동 구간'


def ankle_speed(seq):
    v = np.diff(seq[:, [27, 28]][:, :, [0, 2]], axis=0) * FPS
    return np.linalg.norm(v, axis=-1).mean(1)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "shovel/HmyvWOXlSt0"
    path = os.path.join(E.RAW_KEYPOINTS_DIR, name + ".npz")
    seq_full, vis, det = E.load_raw_keypoints_full(path)
    raw = seq_full[det]
    provided = E.create_height_corrected_target(raw)
    refined = E.refine(seq_full, vis, fps=FPS)[0][det]

    # 이동 구간 판정: 개선본(루트 모션 복원됨)의 발 위치 추정치가 움직이는 속도
    fc = refined[:, FEET][:, :, [0, 2]].mean(1)
    slow = _rolling_median(fc, 3 * FPS)
    move_speed = np.r_[0, np.linalg.norm(np.diff(slow, axis=0), axis=1) * FPS]
    move_speed = np.convolve(move_speed, np.ones(15) / 15, mode="same")
    moving = move_speed > MOVE_THRESH

    sp_p, sp_r = ankle_speed(provided), ankle_speed(refined)
    m = moving[1:]
    res = {}
    for lab, mask in (("stationary", ~m), ("moving", m)):
        a, b = sp_p[mask].mean(), sp_r[mask].mean()
        res[lab] = {"ratio(%)": round(100 * mask.mean(), 1),
                    "foot_slide_provided(m/s)": round(float(a), 3),
                    "foot_slide_refined(m/s)": round(float(b), 3),
                    "reduction(%)": round(float(100 * (1 - b / a)), 1)}
    print(json.dumps(res, indent=2, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(refined)) / FPS
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for a_ in ax:
        a_.fill_between(t, 0, 1, where=moving, transform=a_.get_xaxis_transform(),
                        color="#f0ad4e", alpha=0.25, lw=0, label="moving (assumption broken)")
    ax[0].plot(t, move_speed, color="#555", lw=1)
    ax[0].axhline(MOVE_THRESH, ls="--", color="#999", lw=0.8)
    ax[0].set_ylabel("foot position\nspeed (m/s)")
    ax[0].set_title(f"{name} — where the 'stationary feet' assumption breaks")
    ax[0].legend(loc="upper right", fontsize=8)
    k = 15
    smooth = lambda x: np.convolve(x, np.ones(k) / k, mode="same")
    ax[1].plot(t[1:], smooth(sp_p), color="#d9534f", lw=1, label="Provided")
    ax[1].plot(t[1:], smooth(sp_r), color="#2b7bba", lw=1.3, label="Refined (ours)")
    ax[1].set_ylabel("ankle horiz. speed\n(m/s, 0.5s avg)")
    ax[1].set_xlabel("time (s)")
    s, mv = res["stationary"], res["moving"]
    ax[1].set_title(f"Foot sliding reduction — stationary {s['reduction(%)']:.0f}%  vs  moving {mv['reduction(%)']:.0f}%")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(E.OUT_DIR, name.replace("/", "__") + "_failure_case.png")
    os.makedirs(E.OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=150)
    with open(out.replace(".png", ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved →", out)


if __name__ == "__main__":
    main()
