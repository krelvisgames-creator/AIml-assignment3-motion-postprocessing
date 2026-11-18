"""
rigid_tool.py
-------------
'도구는 휘지 않는다'(강체 제약) 후처리 + 손 방향 정리.

배경
  BlazePose 는 도구를 모른다. 해머 · 삽은 두 손이 '한 자루'를 쥐므로 두 손목 사이 거리가 거의 일정해야 하는데,
  추정 결과는 17~48cm 사이를 오간다(손이 몸에 가려질 때 특히). 그대로 리타게팅하면 손이 벌어졌다 붙었다 하고,
  손에 도구를 붙이면 도구가 손과 따로 논다.

처리 (프레임마다)
  1) 손 점(검지 19/20 · 새끼 17/18 · 엄지 21/22)을 시간축으로 부드럽게(Savitzky-Golay) — 손 방향 잡음 감소.
  2) 그립 간격 G = 클립의 두 손목 거리 중앙값(두 손 쥠 구간). 두 손목을 '가운데는 그대로, 방향은 시간축으로 부드럽게,
     간격은 1초 창 평활 + G±25%'로 다시 놓는다 → 손이 자루를 따라 천천히 미끄러지는 건 남기고, 프레임마다 튀는 잡음만 없앰.
  3) 팔 길이 유지: 손목이 바뀐 만큼 팔꿈치를 2-bone IK(위팔 · 아래팔 길이 고정, 원래 팔꿈치 쪽으로 굽힘)로 다시 놓는다.
     손 점들은 손목이 움직인 만큼 같이 옮긴다.
  4) 가상 도구 관절 2개를 덧붙인다: 33 = 자루 끝(뒷손 뒤 10cm), 34 = 도구 머리(자루 끝에서 도구 길이).
     뒷손 = 클립 전체에서 어깨에 더 가까운 쪽이 많았던 손(프레임마다 바꾸면 도구가 두 손 사이를 오감).

지표: 그립 간격 변동계수(CV), 자루 방향 각속도(°/frame, 떨림), 아래팔 길이 CV(팔 왜곡이 없는지).
"""
import numpy as np
from scipy.signal import savgol_filter

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
HAND = {L_WR: (17, 19, 21), R_WR: (18, 20, 22)}
TOOL_LEN = {"hammer": 0.85, "shovel": 1.05}


def _smooth(x, fps, win_s=0.25, order=2):
    w = max(5, int(round(win_s * fps)) | 1)
    if len(x) <= w:
        return x.copy()
    return savgol_filter(x, w, order, axis=0)


def _two_bone_ik(sh, el, wr_new, l1, l2):
    """어깨 sh 에서 손목 wr_new 로 가는 팔: 위팔 l1 · 아래팔 l2 고정, 굽힘 방향은 원래 팔꿈치 el 쪽."""
    d = wr_new - sh
    dist = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-6)
    reach = l1 + l2 - 1e-4
    dist_c = np.minimum(dist, reach[:, None])
    u = d / dist
    wr = sh + u * dist_c                                      # 닿지 않으면 팔 끝까지만
    # 팔꿈치: 어깨에서 a 만큼 u 방향 + 굽힘 방향으로 h
    a = (l1[:, None] ** 2 - l2[:, None] ** 2 + dist_c ** 2) / (2 * dist_c)
    h = np.sqrt(np.clip(l1[:, None] ** 2 - a ** 2, 0, None))
    pole = el - sh
    pole = pole - (pole * u).sum(1, keepdims=True) * u
    pn = np.linalg.norm(pole, axis=1, keepdims=True)
    fallback = np.cross(u, np.array([0.0, 1.0, 0.0]))
    pole = np.where(pn > 1e-6, pole / pn.clip(1e-6), fallback / np.linalg.norm(fallback, axis=1, keepdims=True).clip(1e-6))
    el_new = sh + u * a + pole * h
    return el_new, wr


def grip_metrics(seq):
    g = np.linalg.norm(seq[:, L_WR] - seq[:, R_WR], axis=1)
    fl = np.linalg.norm(seq[:, L_WR] - seq[:, L_EL], axis=1)
    fr = np.linalg.norm(seq[:, R_WR] - seq[:, R_EL], axis=1)
    ax = seq[:, R_WR] - seq[:, L_WR]
    ax /= np.linalg.norm(ax, axis=1, keepdims=True).clip(1e-6)
    ang = np.degrees(np.arccos(np.clip((ax[1:] * ax[:-1]).sum(1), -1, 1)))
    return {
        "grip_gap_mean_m": float(g.mean()),
        "grip_gap_CV": float(g.std() / g.mean()),
        "handle_angvel_deg_per_frame": float(ang.mean()),
        "forearm_len_CV": float((fl.std() / fl.mean() + fr.std() / fr.mean()) / 2),   # 팔마다 따로(좌우 길이 차이는 빼고)
    }


def apply_rigid_tool(seq, fps=30.0, tool="hammer"):
    """seq (T,33,3) → (T,35,3) : 강체 그립 + 팔 IK + 손 점 정리 + 도구 관절 2개."""
    out = seq.copy()
    T = len(seq)

    # 1) 손 점 시간 평활
    for wr, pts in HAND.items():
        for j in pts:
            out[:, j] = _smooth(out[:, j], fps)

    # 2) 강체 그립
    lw, rw = out[:, L_WR].copy(), out[:, R_WR].copy()
    gap = np.linalg.norm(rw - lw, axis=1)
    two_hand = gap < 0.6
    G = float(np.median(gap[two_hand])) if two_hand.any() else float(np.median(gap))
    mid = (lw + rw) / 2
    ax = rw - lw
    ax /= np.linalg.norm(ax, axis=1, keepdims=True).clip(1e-6)
    ax_s = _smooth(ax, fps, win_s=0.3)                      # 자루 방향은 손목보다 더 부드럽게
    ax_s /= np.linalg.norm(ax_s, axis=1, keepdims=True).clip(1e-6)
    # 손을 자루 따라 미끄러뜨리는 건 실제 동작(느림), 프레임마다 튀는 건 잡음(빠름) → 간격은 1초 창으로 평활 + G 의 ±25% 로 제한
    gap_t = np.clip(_smooth(gap, fps, win_s=1.0), 0.75 * G, 1.25 * G)[:, None]
    lw_t, rw_t = mid - ax_s * gap_t / 2, mid + ax_s * gap_t / 2

    # 3) 팔 길이 유지 IK
    for sh, el, wr, tgt in ((L_SH, L_EL, L_WR, lw_t), (R_SH, R_EL, R_WR, rw_t)):
        l1 = np.full(T, np.median(np.linalg.norm(out[:, el] - out[:, sh], axis=1)))
        l2 = np.full(T, np.median(np.linalg.norm(out[:, wr] - out[:, el], axis=1)))
        el_new, wr_new = _two_bone_ik(out[:, sh], out[:, el], tgt, l1, l2)
        shift = wr_new - out[:, wr]
        out[:, el], out[:, wr] = el_new, wr_new
        for j in HAND[wr]:
            out[:, j] += shift

    # 4) 도구 관절: 뒷손(어깨에 가까운 쪽이 더 많았던 손) → 앞손 방향
    shc = (out[:, L_SH] + out[:, R_SH]) / 2
    left_rear = (np.linalg.norm(out[:, L_WR] - shc, axis=1) < np.linalg.norm(out[:, R_WR] - shc, axis=1)).mean() >= 0.5
    rear, lead = (out[:, L_WR], out[:, R_WR]) if left_rear else (out[:, R_WR], out[:, L_WR])
    hdir = lead - rear
    hdir /= np.linalg.norm(hdir, axis=1, keepdims=True).clip(1e-6)
    butt = rear - hdir * 0.10
    head = butt + hdir * TOOL_LEN.get(tool, 0.9)
    full = np.concatenate([out, butt[:, None], head[:, None]], axis=1)
    info = {"grip_len_G_m": round(G, 3), "rear_hand": "left" if left_rear else "right"}
    return full, info
