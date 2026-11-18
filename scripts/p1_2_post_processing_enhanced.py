"""
p1_2_post_processing_enhanced.py
--------------------------------
[Assignment 3] 3D Motion Data Post-Processing (Enhanced)
Author: Junha Yoo (20242134)

원본 p1_2_post_processing.py 를 기반으로, BlazePose 3D 키포인트의 품질 문제를
'애니메이터식 클린업' 파이프라인으로 개선한다. (세부 구현: animator_cleanup.py)

  RAW (.npz)
   │ 1) load_raw_keypoints_full : frame 번호 기준 배치 → 검출 실패 프레임을 NaN 으로 보존(시간 압축 방지)
   │ 2) remove_outliers / fill_gaps : 속도 스파이크 관절 제거 후 시간축 보간
   │ 3) normalize_bone_lengths  : 프레임마다 늘었다 줄었다 하는 뼈 길이를 시퀀스 중앙값으로 고정
   │ 4) detect_impact_keys      : 손목 최저점 = 해머/삽이 땅에 닿는 순간 → 반드시 남길 '임팩트 키'
   │ 5) key_reduce + inbetween_pchip : RDP 키 리덕션 후 PCHIP 인비트윈 (떨림 제거, 타격감 보존)
   │ 6) anchor_feet             : 발 미끄러짐 제거 + 루트 모션 복원 (골반 고정 좌표계의 한계 해결)
   │ 7) plant_feet_ik           : 각 발을 플랜트 위치에 고정, 2-bone IK 로 무릎 재계산
   │ 8) ground_constant         : 바닥 높이 = 0 (상수 1회 보정 → 몸 전체 상하 떨림 없음)
   ▼
  REFINED (.npy / Unity .csv) + metrics.json + 4-panel 비교 그래프(.png)

원본 코드 대비 변경점
  - load_raw_keypoints : 검출 실패 프레임이 빠지면서 시간이 압축되던 문제 수정
  - create_height_corrected_target(제공 방법) 은 비교 기준(baseline)으로 그대로 유지.
    매 프레임 발끝 최저점을 0으로 맞추는 방식이라 (a) 발 노이즈가 몸 전체 상하 떨림으로 전달되고
    (b) 수평 방향 발 미끄러짐은 해결하지 못함 → 6)~8) 로 대체
  - --method savgol / oneeuro : 필터 스무딩 버전(비교 실험용)

가정/한계: 발이 거의 고정된 '제자리 동작'(해머, 삽질) 전제. 걷기·방향 전환에는 부적합.
"""

import os
import sys
import json
import argparse
from glob import glob

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")

N_JOINTS = 33

# BlazePose skeleton (viser_test.POSE_CONNECTIONS 와 동일)
POSE_CONNECTIONS = [
    (11, 12), (12, 14), (14, 16), (11, 13), (13, 15),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]

# 뼈 길이 정규화용 kinematic tree (parent → child). 루트는 골반 중심(23,24 중점)
BONE_TREE = [
    (23, 11), (24, 12),            # hip → shoulder (몸통)
    (11, 13), (13, 15),            # left arm
    (12, 14), (14, 16),            # right arm
    (23, 25), (25, 27), (27, 29), (29, 31),   # left leg
    (24, 26), (26, 28), (28, 30), (30, 32),   # right leg
]
# 트리에 없는 관절(얼굴/손가락)은 붙어 있는 관절을 따라 강체 이동
FOLLOWERS = {
    "head": (list(range(0, 11)), (11, 12)),   # 얼굴 → 양 어깨 중점
    15: [17, 19, 21],                          # 왼손가락 → 왼손목
    16: [18, 20, 22],                          # 오른손가락 → 오른손목
}
FOOT_JOINTS = [29, 30, 31, 32]   # heel L/R, foot index L/R


# =====================================================================
# 1) Load  (원본 수정: 검출 실패 프레임 보존)
# =====================================================================
def load_raw_keypoints_full(npz_path):
    """
    Returns
      seq : (T,33,3)  검출 실패 프레임은 NaN
      vis : (T,33)    visibility (실패 프레임은 0)
      detected : (T,) bool
    """
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame", "landmark", "x", "y", "z", "visibility"])
    df = df.astype({"frame": int, "landmark": int, "x": float, "y": float,
                    "z": float, "visibility": float})

    T = int(df["frame"].max()) + 1
    seq = np.full((T, N_JOINTS, 3), np.nan)
    vis = np.zeros((T, N_JOINTS))
    f = df["frame"].values
    j = df["landmark"].values
    seq[f, j] = df[["x", "y", "z"]].values
    vis[f, j] = df["visibility"].values
    detected = ~np.isnan(seq).any(axis=(1, 2))
    return seq, vis, detected


def load_raw_keypoints(npz_path):
    """원본 방식 (비교용): 검출된 프레임만 이어붙임 → (T',33,3)"""
    seq, _, detected = load_raw_keypoints_full(npz_path)
    return seq[detected]


# =====================================================================
# 2) Outlier removal (velocity spike)
# =====================================================================
def remove_outliers(seq, vis=None, k=6.0, vis_thresh=0.3):
    """
    관절별 프레임 간 이동량이 (median + k*MAD)를 넘으면 튐(spike)으로 보고 NaN 처리.
    visibility가 매우 낮은 관절도 NaN 처리 (가려진 관절의 엉뚱한 추정값 제거).
    """
    out = seq.copy()
    vel = np.linalg.norm(np.diff(seq, axis=0), axis=-1)          # (T-1,33)
    med = np.nanmedian(vel, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(vel - med), axis=0, keepdims=True) + 1e-6
    spike = vel > med + k * 1.4826 * mad                          # (T-1,33)
    # 들어갈 때와 나올 때 모두 튀면 그 프레임이 outlier
    bad = np.zeros(seq.shape[:2], dtype=bool)
    bad[1:-1] = spike[:-1] & spike[1:]
    if vis is not None:
        bad |= (vis < vis_thresh) & ~np.isnan(seq[..., 0])
    out[bad] = np.nan
    return out, int(bad.sum())


# =====================================================================
# 3) Gap filling (linear interpolation along time)
# =====================================================================
def fill_gaps(seq):
    out = seq.copy()
    T = seq.shape[0]
    t = np.arange(T)
    for j in range(seq.shape[1]):
        for c in range(3):
            y = out[:, j, c]
            ok = ~np.isnan(y)
            if ok.sum() == 0:
                out[:, j, c] = 0.0
            elif ok.sum() < T:
                out[:, j, c] = np.interp(t, t[ok], y[ok])   # 양 끝은 가장 가까운 값 유지
    return out


# =====================================================================
# 4-a) Savitzky-Golay smoothing
# =====================================================================
def smooth_savgol(seq, window=9, poly=3):
    T = seq.shape[0]
    if T < 5:
        return seq.copy()
    w = min(window, T if T % 2 == 1 else T - 1)
    p = min(poly, w - 1)
    return savgol_filter(seq, window_length=w, polyorder=p, axis=0, mode="interp")


# =====================================================================
# 4-b) One Euro filter (Casiez et al., CHI 2012) — 느린 동작은 강하게, 빠른 동작은 약하게
# =====================================================================
def smooth_one_euro(seq, fps=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)

    out = np.empty_like(seq)
    x_prev = seq[0].copy()
    dx_prev = np.zeros_like(seq[0])
    out[0] = x_prev
    a_d = alpha(d_cutoff)
    for t in range(1, seq.shape[0]):
        x = seq[t]
        dx = (x - x_prev) * fps
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        tau = 1.0 / (2 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau * fps)
        x_hat = a * x + (1 - a) * x_prev
        out[t] = x_hat
        x_prev, dx_prev = x_hat, dx_hat
    return out


# =====================================================================
# 5) Bone length normalization
# =====================================================================
def normalize_bone_lengths(seq):
    """
    BlazePose는 프레임마다 뼈 길이를 독립 추정 → 팔다리가 늘었다 줄었다 함.
    시퀀스 전체의 중앙값 뼈 길이로 고정하고, 방향(direction)은 그대로 유지.
    """
    out = seq.copy()
    pelvis = (seq[:, 23] + seq[:, 24]) / 2.0

    # 골반 폭 고정 (루트 기준 좌우 대칭)
    hip_dir = seq[:, 24] - seq[:, 23]
    hip_len = np.linalg.norm(hip_dir, axis=-1, keepdims=True) + 1e-8
    hip_target = np.median(hip_len)
    out[:, 23] = pelvis - hip_dir / hip_len * hip_target / 2
    out[:, 24] = pelvis + hip_dir / hip_len * hip_target / 2

    for parent, child in BONE_TREE:
        d = seq[:, child] - seq[:, parent]
        L = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
        target = np.median(L)
        out[:, child] = out[:, parent] + d / L * target

    # 트리 밖 관절은 기준 관절의 이동량만큼 따라 이동
    head_idx, (a, b) = FOLLOWERS["head"]
    delta = (out[:, a] + out[:, b]) / 2 - (seq[:, a] + seq[:, b]) / 2
    out[:, head_idx] = seq[:, head_idx] + delta[:, None]
    for anchor in (15, 16):
        delta = out[:, anchor] - seq[:, anchor]
        out[:, FOLLOWERS[anchor]] = seq[:, FOLLOWERS[anchor]] + delta[:, None]
    return out


# =====================================================================
# 6) Ground alignment (원본 create_height_corrected_target 개선)
# =====================================================================
def create_height_corrected_target(raw_seq):
    """원본 코드 그대로 (비교용)"""
    seq = raw_seq.copy()
    foot_y = np.minimum(seq[:, 31, 1], seq[:, 32, 1])
    seq[:, :, 1] -= foot_y[:, None]
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
    return seq


def ground_alignment(seq, smooth_window=15):
    """
    - 발끝뿐 아니라 뒤꿈치까지 포함한 4개 발 관절의 최저점을 접지 높이로 사용
    - 접지 오프셋 자체를 시간축으로 평활해 '전신이 위아래로 떨리는' 현상 제거
    - clamp 대신 전체 평행이동만 수행 → 자세 형태 보존
    """
    out = seq.copy()
    ground = np.min(seq[:, FOOT_JOINTS, 1], axis=1)               # (T,)
    if len(ground) >= 5:
        w = min(smooth_window, len(ground) if len(ground) % 2 else len(ground) - 1)
        ground = savgol_filter(ground, w, min(2, w - 1), mode="interp")
    out[:, :, 1] -= ground[:, None]
    return out


# =====================================================================
# Evaluation metrics
# =====================================================================
def bone_length_cv(seq):
    """뼈별 길이 변동계수(std/mean) 평균 — 낮을수록 뼈 길이가 일정"""
    cvs = []
    for a, b in POSE_CONNECTIONS:
        L = np.linalg.norm(seq[:, a] - seq[:, b], axis=-1)
        cvs.append(np.std(L) / (np.mean(L) + 1e-8))
    return float(np.mean(cvs))


def jitter(seq, fps=30.0):
    """평균 가속도 크기 (m/s^2) — 낮을수록 떨림이 적음"""
    acc = np.diff(seq, n=2, axis=0) * fps * fps
    return float(np.mean(np.linalg.norm(acc, axis=-1)))


def jerk(seq, fps=30.0):
    """평균 jerk 크기 (m/s^3)"""
    j = np.diff(seq, n=3, axis=0) * fps ** 3
    return float(np.mean(np.linalg.norm(j, axis=-1)))


def pelvis_vertical_jitter(seq):
    """골반 높이의 프레임 간 변화량 평균 (m) — 접지 보정이 전신을 위아래로 흔드는 정도"""
    pelvis_y = (seq[:, 23, 1] + seq[:, 24, 1]) / 2
    return float(np.mean(np.abs(np.diff(pelvis_y))))


def evaluate(seq, fps=30.0):
    from animator_cleanup import foot_slide_speed, root_travel
    return {
        "jitter_acc(m/s^2)": round(jitter(seq, fps), 3),
        "jerk(m/s^3)": round(jerk(seq, fps), 1),
        "bone_length_CV": round(bone_length_cv(seq), 4),
        "foot_slide(m/s)": round(foot_slide_speed(seq, fps), 4),
        "pelvis_vert_jitter(m/frame)": round(pelvis_vertical_jitter(seq), 5),
        "root_travel(m)": round(root_travel(seq), 3),
    }


# =====================================================================
# Plot
# =====================================================================
C_RAW, C_BASE, C_REF = "#9aa0a6", "#d9534f", "#2b7bba"


def save_comparison_plot(raw, baseline, refined, info, path, fps=30.0, title=""):
    """
    4-panel 비교 그래프
      (a) 발목 수평 궤적(위에서 본 모습)  — 발 미끄러짐
      (b) 손목 높이 + 임팩트 키 + 키프레임 — 키 리덕션/인비트윈
      (c) 발목 수평 속도                   — 발 미끄러짐 시간축
      (d) 정강이 뼈 길이                    — 뼈 길이 일관성
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(raw.shape[0]) / fps
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2])

    # (a) top-down ankle trajectories
    ax = fig.add_subplot(gs[:, 0])
    for seq, c, lab in ((baseline, C_BASE, "Provided (height corr.)"), (refined, C_REF, "Refined (ours)")):
        for j, ls in ((27, "-"), (28, "--")):
            ax.plot(seq[:, j, 0], seq[:, j, 2], ls, color=c, lw=0.8, alpha=0.8,
                    label=f"{lab} {'L' if j == 27 else 'R'} ankle")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("(a) Ankle path, top view\n(should stay in place)")
    ax.legend(fontsize=7, loc="upper right")

    # (b) wrist height + keys  — 바닥 정렬 방식이 서로 달라서, 둘 다 '골반 기준 높이'로 비교
    def wrist_rel(q):
        return q[:, [15, 16], 1].mean(1) - (q[:, 23, 1] + q[:, 24, 1]) / 2
    ax = fig.add_subplot(gs[0, 1])
    wb = wrist_rel(raw)
    wr = wrist_rel(refined)
    ax.plot(t, wb, color=C_BASE, lw=0.8, label="Raw / Provided")
    ax.plot(t, wr, color=C_REF, lw=1.4, label="Refined (PCHIP in-between)")
    k = info.get("keys_on_detected")
    if k is not None and len(k):
        ax.plot(t[k], wr[k], "o", ms=2.5, color="#333", label=f"keys ({len(k)})")
    imp = info.get("impacts_on_detected")
    if imp is not None and len(imp):
        ax.plot(t[imp], wr[imp], "v", ms=7, color="#f0ad4e", label=f"impact keys ({len(imp)})")
    ax.set_ylabel("wrist height\nrel. to pelvis (m)")
    ax.set_title("(b) Wrist height — key reduction & impact keys")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # (c) ankle horizontal speed
    ax = fig.add_subplot(gs[1, 1], sharex=None)
    def spd(s):
        v = np.diff(s[:, [27, 28]][:, :, [0, 2]], axis=0) * fps
        return np.linalg.norm(v, axis=-1).mean(1)
    ax.plot(t[1:], spd(baseline), color=C_BASE, lw=0.7, label=f"Provided  mean {spd(baseline).mean():.3f} m/s")
    ax.plot(t[1:], spd(refined), color=C_REF, lw=1.2, label=f"Refined   mean {spd(refined).mean():.3f} m/s")
    ax.set_ylabel("ankle horiz. speed (m/s)")
    ax.set_title("(c) Foot sliding")
    ax.legend(fontsize=7, loc="upper right")

    # (d) shin length
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, np.linalg.norm(baseline[:, 25] - baseline[:, 27], axis=-1), color=C_BASE, lw=0.7, label="Provided")
    ax.plot(t, np.linalg.norm(refined[:, 25] - refined[:, 27], axis=-1), color=C_REF, lw=1.4, label="Refined")
    ax.set_ylabel("L shin length (m)")
    ax.set_xlabel("time (s)")
    ax.set_title("(d) Bone length consistency")
    ax.legend(fontsize=7, loc="upper right")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")


# =====================================================================
# Unity CSV (원본과 동일)
# =====================================================================
def export_unity_csv(motion, output_path):
    T, n_landmarks, _ = motion.shape
    frames = np.repeat(np.arange(T), n_landmarks)
    lms = np.tile(np.arange(n_landmarks), T)
    xyz = motion.reshape(-1, 3)
    df = pd.DataFrame({
        "frame": frames, "landmark": lms,
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "visibility": 1.0,
    })
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Unity CSV saved: {output_path}  (frames={T}, rows={len(df)})")


# =====================================================================
# Full pipeline
# =====================================================================
def refine(seq_full, vis, fps=30.0, method="animator", key_tol=0.04):
    """
    method="animator" (기본, 최종안)
      검출실패/튐 복원 → 뼈 길이 고정 → [임팩트 키 검출 → 키 리덕션 → PCHIP 인비트윈]
      → 뼈 길이 재고정 → [발 앵커링(루트 모션) → 발별 플랜트 + 2-bone IK] → 바닥 정렬
    method="savgol" / "oneeuro" : 이전 버전(필터 스무딩) — 비교용
    """
    from animator_cleanup import (detect_impact_keys, key_reduce, inbetween_pchip,
                                  anchor_feet, plant_feet_ik, ground_constant)
    log = {}
    log["missing_frames"] = int(np.isnan(seq_full).any(axis=(1, 2)).sum())
    seq, n_out = remove_outliers(seq_full, vis=None)
    log["outlier_joints_removed"] = n_out
    seq = fill_gaps(seq)

    if method in ("savgol", "oneeuro"):
        seq = smooth_one_euro(seq, fps=fps) if method == "oneeuro" else smooth_savgol(seq)
        seq = normalize_bone_lengths(seq)
        seq = ground_alignment(seq)
        return seq, log

    # --- animator pipeline ---
    seq = normalize_bone_lengths(seq)

    impacts, _ = detect_impact_keys(seq, fps)
    keys = key_reduce(seq, tol=key_tol, forced_keys=impacts)
    seq_kr = inbetween_pchip(seq, keys)
    log["key_reconstruction_rmse(m)"] = round(float(np.sqrt(((seq_kr - seq) ** 2).sum(-1).mean())), 4)
    seq = normalize_bone_lengths(seq_kr)

    seq, root_off = anchor_feet(seq, fps)
    seq, lock_ratio = plant_feet_ik(seq, fps)
    seq = ground_constant(seq)

    T = seq.shape[0]
    log["impact_keys"] = int(len(impacts))
    if len(impacts) > 1:
        log["swing_period(s)"] = round(float(np.median(np.diff(impacts)) / fps), 2)
    log["keys"] = int(len(keys))
    log["key_ratio(%)"] = round(100.0 * len(keys) / T, 1)
    log["foot_lock_ratio(L,R)"] = [round(r, 2) for r in lock_ratio]
    log["_keys"] = keys
    log["_impacts"] = impacts
    return seq, log


def _to_detected_index(idx, detected):
    """전체 프레임 인덱스 → 검출 프레임만 남긴 배열에서의 인덱스"""
    pos = np.cumsum(detected) - 1
    idx = np.asarray(idx, dtype=int)
    idx = idx[detected[idx]]
    return pos[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--method", choices=["animator", "savgol", "oneeuro"], default="animator")
    ap.add_argument("--key_tol", type=float, default=0.04, help="키 리덕션 허용 오차(m)")
    ap.add_argument("--input_dir", default=RAW_KEYPOINTS_DIR)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    # raw_keypoints/*.npz 와 raw_keypoints/<class>/*.npz 모두 처리
    paths = sorted(glob(os.path.join(args.input_dir, "**", "*.npz"), recursive=True))
    print(f"Found {len(paths)} raw keypoint files")

    report = {}
    for path in paths:
        rel = os.path.relpath(path, args.input_dir)
        base = os.path.splitext(rel)[0].replace(os.sep, "__")     # 예: hammer__wFXKSWzuHe0
        print(f"\n--- {base}")
        seq_full, vis, detected = load_raw_keypoints_full(path)

        raw_seq = seq_full[detected]                               # 원본 (검출 프레임만)
        baseline = create_height_corrected_target(raw_seq)         # 제공된 후처리
        refined, log = refine(seq_full, vis, fps=args.fps, method=args.method, key_tol=args.key_tol)

        np.save(os.path.join(OUT_DIR, f"{base}_raw.npy"), raw_seq)
        np.save(os.path.join(OUT_DIR, f"{base}_target.npy"), baseline)
        np.save(os.path.join(OUT_DIR, f"{base}_refined.npy"), refined)
        export_unity_csv(baseline, os.path.join(OUT_DIR, f"{base}_target.csv"))
        export_unity_csv(refined, os.path.join(OUT_DIR, f"{base}_refined.csv"))

        # 비교는 검출된 프레임 위치에서 (프레임 수를 맞추기 위해)
        refined_d = refined[detected]
        info = {}
        if "_keys" in log:
            info["keys_on_detected"] = _to_detected_index(log.pop("_keys"), detected)
            info["impacts_on_detected"] = _to_detected_index(log.pop("_impacts"), detected)
        report[base] = {
            "frames_total": int(len(detected)),
            **log,
            "raw": evaluate(raw_seq, args.fps),
            "provided": evaluate(baseline, args.fps),
            "refined": evaluate(refined_d, args.fps),
        }
        save_comparison_plot(raw_seq, baseline, refined_d, info,
                             os.path.join(OUT_DIR, f"{base}_compare.png"),
                             fps=args.fps, title=base)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    print("\nDone processing all keypoint files.")


if __name__ == "__main__":
    main()
