"""
animator_cleanup.py
-------------------
'애니메이터식' 모션 정리 모듈 — p1_2_post_processing_enhanced.py 에서 사용
Author: Junha Yoo (20242134)

모캡 클린업 현장에서 애니메이터가 손으로 하던 작업을 자동화한다.

  1) detect_impact_keys   : 손목 최저점 = 해머/삽이 땅에 닿는 순간 → '반드시 남길 키'
  2) key_reduce           : 포즈 전체 기준 RDP 키 리덕션 (임팩트 키 강제 유지)
  3) inbetween_pchip      : 남은 키 사이를 PCHIP 스플라인으로 인비트윈
                            (단조 보간 = 오버슈트 없음, Maya 'clamped/auto tangent' 와 유사)
  4) anchor_feet          : 발 미끄러짐 제거 + 루트 모션 복원
       - BlazePose world landmark 는 골반이 원점(0,0,0)에 고정 → 실제로는 골반이 움직인 것이
         '발이 미끄러지는 것'으로 나타남
       - 양발 중심을 '느린 위치(rolling median)'에 고정하도록 몸 전체를 평행이동 → 골반이 움직이게 됨
  5) plant_feet_ik        : 각 발을 개별 고정 위치로 2-bone IK (고관절-무릎-발목) — 플랜트 키 + IK 고정

가정(한계): 해머·삽질처럼 발이 거의 고정된 '제자리 동작'. 걷기/방향 전환은 rolling median
윈도우보다 느린 이동만 보존되고, 빠른 스텝은 미끄러지듯 처리된다.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter, uniform_filter1d

L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32
L_FOOT = [L_ANKLE, L_HEEL, L_TOE]
R_FOOT = [R_ANKLE, R_HEEL, R_TOE]
FEET = L_FOOT + R_FOOT
# 키 판정에 쓰는 '몸의 큰 관절' (얼굴·손가락 노이즈 때문에 키가 남발되지 않게)
CORE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


# ---------------------------------------------------------------------
# 1) Impact key detection
# ---------------------------------------------------------------------
def detect_impact_keys(seq, fps=30.0, min_period=0.4):
    """
    양 손목 평균 높이(y)의 국소 최저점을 임팩트 프레임으로 검출.
    prominence 는 손목 높이 분포(5~95 percentile)의 25% 이상 내려간 골만 인정.
    Returns: impact frame indices (np.ndarray), wrist height curve (smoothed)
    """
    wrist_y = seq[:, [15, 16], 1].mean(axis=1)
    if len(wrist_y) >= 9:
        wrist_y = savgol_filter(wrist_y, 9, 2)
    lo, hi = np.percentile(wrist_y, [5, 95])
    prom = max(0.05, 0.25 * (hi - lo))
    peaks, _ = find_peaks(-wrist_y, prominence=prom, distance=max(1, int(min_period * fps)))
    return peaks, wrist_y


# ---------------------------------------------------------------------
# 2) Key reduction (RDP on full pose)
# ---------------------------------------------------------------------
def key_reduce(seq, tol=0.04, forced_keys=(), joints=CORE):
    """
    포즈(몸의 큰 관절 12개)를 하나의 곡선으로 보고 Ramer–Douglas–Peucker 로 키 프레임 선택.
    두 키 사이를 선형 보간했을 때 '어느 큰 관절이든' tol(m) 이상 벗어나는 프레임이 있으면 그 프레임을 키로 추가.
    forced_keys (임팩트 등) 는 무조건 키로 유지. 키는 포즈 단위(전 관절 공통)로 찍힌다.
    """
    seq = seq[:, joints]
    T = seq.shape[0]
    keys = {0, T - 1, *[int(k) for k in forced_keys if 0 <= k < T]}
    anchors = sorted(keys)
    stack = list(zip(anchors[:-1], anchors[1:]))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = np.arange(a + 1, b)
        w = ((t - a) / (b - a))[:, None, None]
        lin = seq[a] * (1 - w) + seq[b] * w
        err = np.linalg.norm(seq[a + 1:b] - lin, axis=-1).max(axis=1)   # 프레임별 최대 관절 오차
        i = int(np.argmax(err))
        if err[i] > tol:
            m = a + 1 + i
            keys.add(m)
            stack.append((a, m))
            stack.append((m, b))
    return np.array(sorted(keys))



# ---------------------------------------------------------------------
# 3) In-betweening
# ---------------------------------------------------------------------
def inbetween_pchip(seq, keys):
    """키 프레임 사이를 PCHIP(단조 3차 에르미트) 로 보간 → 오버슈트 없는 부드러운 인비트윈"""
    T = seq.shape[0]
    flat = seq[keys].reshape(len(keys), -1)
    f = PchipInterpolator(keys, flat, axis=0)
    return f(np.arange(T)).reshape(seq.shape)


# ---------------------------------------------------------------------
# 4) Foot anchoring (root motion recovery)
# ---------------------------------------------------------------------
def _rolling_median(x, win):
    """(T, D) 시계열의 시간축 rolling median (가장자리는 nearest)"""
    win = max(3, int(win) | 1)
    return median_filter(x, size=(win, 1), mode="nearest")


def anchor_feet(seq, fps=30.0, anchor_window=3.0):
    """
    양발 6점(발목·뒤꿈치·발끝)의 수평 중심(x,z)을 'anchor_window 초 rolling median' 위치에 고정.
    고정에 필요한 이동량을 몸 전체에 적용 → 발은 제자리, 골반은 실제처럼 이동(루트 모션).
    Returns: anchored seq, root offset (T,3)
    """
    center = seq[:, FEET][:, :, [0, 2]].mean(axis=1)              # (T,2)
    # 발 좌표의 프레임 단위 노이즈가 몸 전체 이동량으로 옮겨가지 않도록 중심을 먼저 평활
    w = min(15, len(center) - (1 - len(center) % 2))
    if w >= 5:
        center = savgol_filter(center, w, 2, axis=0)
    anchor = _rolling_median(center, anchor_window * fps)         # 느린 위치 (실제 발 이동만 남음)
    off_xz = anchor - center
    off = np.zeros((seq.shape[0], 3))
    off[:, 0], off[:, 2] = off_xz[:, 0], off_xz[:, 1]
    return seq + off[:, None, :], off


# ---------------------------------------------------------------------
# 5) Per-foot plant + 2-bone IK
# ---------------------------------------------------------------------
def _two_bone_ik(hip, knee, target, l1, l2):
    """hip 고정, ankle 을 target 으로. knee 는 원래 무릎 방향(pole)을 유지. 모두 (T,3)"""
    d_vec = target - hip
    d = np.linalg.norm(d_vec, axis=-1, keepdims=True)
    d = np.clip(d, 1e-6, (l1 + l2) * 0.999)                      # 다리 길이 초과 방지
    u = d_vec / np.linalg.norm(d_vec, axis=-1, keepdims=True).clip(1e-6)
    target = hip + u * d
    a = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(np.clip(l1 ** 2 - a ** 2, 0, None))
    p = knee - hip
    v = p - (p * u).sum(-1, keepdims=True) * u
    v = v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-6)
    new_knee = hip + u * a + v * h
    return new_knee, target


def plant_feet_ik(seq, fps=30.0, plant_window=1.0, release=(0.06, 0.16)):
    """
    각 발을 'plant_window 초 rolling median' 위치에 고정(플랜트)하고 2-bone IK 로 무릎을 다시 계산.
    발이 고정 위치에서 release[0]~release[1] m 이상 벗어나면 '발을 뗀 것'으로 보고 고정을 서서히 해제.
    """
    out = seq.copy()
    lock_ratio = []
    for hip, knee, foot in ((L_HIP, L_KNEE, L_FOOT), (R_HIP, R_KNEE, R_FOOT)):
        ankle = foot[0]
        fc = seq[:, foot].mean(axis=1)                            # 발 중심 (T,3)
        planted = _rolling_median(fc, plant_window * fps)
        dev = np.linalg.norm(fc - planted, axis=-1)
        w = np.clip(1 - (dev - release[0]) / (release[1] - release[0]), 0, 1)
        w = uniform_filter1d(w, size=max(3, int(0.2 * fps)), mode="nearest")   # 고정 on/off 를 부드럽게
        lock_ratio.append(float((w > 0.5).mean()))

        delta = (planted - fc) * w[:, None]                       # 발을 옮길 양
        target = seq[:, ankle] + delta
        l1 = np.median(np.linalg.norm(seq[:, knee] - seq[:, hip], axis=-1))
        l2 = np.median(np.linalg.norm(seq[:, ankle] - seq[:, knee], axis=-1))
        new_knee, new_ankle = _two_bone_ik(seq[:, hip], seq[:, knee], target, l1, l2)
        shift = new_ankle - seq[:, ankle]
        out[:, knee] = new_knee
        for j in foot:                                             # 발목·뒤꿈치·발끝은 강체 이동
            out[:, j] = seq[:, j] + shift
    return out, lock_ratio


def ground_constant(seq):
    """발이 고정된 뒤에는 접지 높이가 거의 일정 → 시퀀스 전체 중앙값 한 번만 빼서 바닥=0"""
    g = np.median(seq[:, FEET, 1].min(axis=1))
    out = seq.copy()
    out[:, :, 1] -= g
    return out


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def foot_slide_speed(seq, fps=30.0):
    """양 발목 수평 속도 평균 (m/s) — 제자리 동작에서 높으면 '발 미끄러짐'"""
    v = np.diff(seq[:, [L_ANKLE, R_ANKLE]][:, :, [0, 2]], axis=0) * fps
    return float(np.linalg.norm(v, axis=-1).mean())


def root_travel(seq):
    """골반 중심 수평 이동 범위 (m) — 복원된 루트 모션 크기"""
    p = (seq[:, L_HIP] + seq[:, R_HIP]) / 2
    return float(np.linalg.norm(np.ptp(p[:, [0, 2]], axis=0)))
