"""Concept figures for the overview slides: the dipole anomaly of a buried
body at Korea's inclination and what RTP does to it; manned vs drone
sampling of the same field; the drone-specific noise sources."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in font_manager.findSystemFonts():
    if "malgun" in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
FIG = Path(__file__).resolve().parent / "fig"
FIG.mkdir(exist_ok=True)
DPI = 200


def dipole_field(x, y, z, m_dir, obs_z=0.0, moment=1.0):
    """Total-field anomaly (projection on the inducing direction) of a point
    dipole at (0, 0, z) with magnetisation direction m_dir, observed on a
    plane at obs_z. Units arbitrary."""
    rx, ry = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rz = np.full_like(rx, obs_z - z)
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    rhat = np.stack([rx, ry, rz], -1) / r[..., None]
    m = np.asarray(m_dir) / np.linalg.norm(m_dir)
    b = moment * (3 * rhat * (rhat @ m)[..., None] - m) / r[..., None] ** 3
    return b @ m


def field_dir(inc_deg, dec_deg=0.0):
    i, d = np.radians(inc_deg), np.radians(dec_deg)
    return np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), -np.sin(i)])   # x east, y north, z up


# 1. anomaly map of a body at inclination 53 deg (Korea) vs pole
x = np.linspace(-150, 150, 301); X, Y = np.meshgrid(x, x)
f_kor, f_pole = field_dir(53), field_dir(90)
a_kor = dipole_field(X, Y, -40, f_kor, obs_z=30)
a_pole = dipole_field(X, Y, -40, f_pole, obs_z=30)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, a, t in zip(axes[:2], (a_kor, a_pole), ("관측 이상 (복각 53°, 한국)", "극자화 변환 후 (복각 90°)")):
    m = np.abs(a).max()
    ax.imshow(a, extent=(-150, 150, -150, 150), cmap="RdBu_r", vmin=-m, vmax=m, origin="lower")
    ax.contour(X, Y, a, levels=10, colors="k", linewidths=0.3)
    ax.plot(0, 0, "k+", ms=12, mew=2); ax.set_title(t, fontsize=12); ax.set_xlabel("동 (m)"); ax.set_ylabel("북 (m)")
ax = axes[2]
ax.plot(x, a_kor[150], color="#b42318", lw=2, label="관측 (복각 53°): 남쪽 음, 북쪽 양")
ax.plot(x, a_pole[150], color="#1f4e79", lw=2, label="RTP 후: 물체 바로 위에 최대")
ax.axvline(0, color="k", ls="--", lw=0.8); ax.set_xlabel("남북 거리 (m)"); ax.set_ylabel("이상 (임의 단위)")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_title("남북 단면", fontsize=12)
fig.suptitle("자화된 물체 하나의 자기이상 — 극자화 변환(RTP)이 왜 필요한가", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "concept_rtp.png", dpi=DPI); plt.close(fig)

# 2. manned vs drone sampling of a field with 50 m-scale sources
rng = np.random.default_rng(3)
x = np.linspace(0, 2000, 401); X, Y = np.meshgrid(x, x)
field = np.zeros_like(X)
for _ in range(14):
    cx, cy, z, s = rng.uniform(200, 1800), rng.uniform(200, 1800), rng.uniform(30, 120), rng.choice([-1, 1])
    field += s * dipole_field(X - cx, Y - cy, -z, f_kor, obs_z=0, moment=z**3 * 2)
field += 0.25 * field.max() * dipole_field(X - 1000, Y - 1000, -600, f_kor, obs_z=0, moment=600**3 * 6)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
m = np.percentile(np.abs(field), 99)
axes[0].imshow(field, extent=(0, 2, 0, 2), cmap="RdBu_r", vmin=-m, vmax=m, origin="lower"); axes[0].set_title("지하의 실제 장 (가상)", fontsize=12)
for ax, (spacing, alt, name) in zip(axes[1:], [(400, 300, "유인 항공: 측선 400 m, 고도 300 m"), (25, 40, "드론: 측선 25 m, 고도 40 m")]):
    xs = np.arange(spacing / 2, 2000, spacing)
    up = np.zeros_like(X)
    # crude upward continuation by Gaussian blur scaled with altitude
    from scipy.ndimage import gaussian_filter
    up = gaussian_filter(field, sigma=alt / 5.0 / 5.0 * 1.0 + alt / 25.0)
    samp = np.full_like(X, np.nan)
    for xx in xs:
        j = int(xx / 5)
        samp[:, j] = up[:, j]
    from scipy.interpolate import griddata
    jj = (xs / 5).astype(int)
    pts = np.array([(X[i, j], Y[i, j]) for j in jj for i in range(0, 401, 4)])
    vals = np.array([up[i, j] for j in jj for i in range(0, 401, 4)])
    gi = griddata(pts, vals, (X, Y), method="linear")
    ax.imshow(gi, extent=(0, 2, 0, 2), cmap="RdBu_r", vmin=-m, vmax=m, origin="lower")
    for xx in xs:
        ax.axvline(xx / 1000, color="k", lw=0.4, alpha=0.5)
    ax.set_title(name, fontsize=12)
for ax in axes:
    ax.set_xlabel("km"); ax.set_ylabel("km")
fig.suptitle("같은 땅을 유인 항공과 드론이 어떻게 다르게 보는가 — 측선 간격과 고도가 해상도를 정한다", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "concept_sampling.png", dpi=DPI); plt.close(fig)

# 3. drone-specific noise: heading effect + sway + altitude variation on a synthetic line set
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
t = np.linspace(0, 120, 1200)
truth = 30 * np.exp(-((t - 60) / 12) ** 2)
axes[0].plot(t, truth + 6 * np.sin(2 * np.pi * 0.7 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.05 * t)) + rng.normal(0, 1.2, t.size), color="#999", lw=0.8, label="측정 (센서 흔들림 + 잡음)")
axes[0].plot(t, truth, color="#b42318", lw=1.5, label="실제 이상"); axes[0].legend(fontsize=9); axes[0].set_title("센서 흔들림(스웨이)·고주파 잡음", fontsize=12); axes[0].set_xlabel("초")
lines = np.arange(10)
head = np.where(lines % 2 == 0, 8, -8)
yy = np.linspace(0, 1, 60)
img = np.array([np.full(60, h) + 20 * np.exp(-((yy - 0.5) / 0.15) ** 2) for h in head]).T
axes[1].imshow(img, aspect="auto", cmap="RdBu_r", vmin=-25, vmax=25, extent=(0, 10, 0, 1)); axes[1].set_title("헤딩 오차: 왕복 측선마다 ±8 nT 어긋남 → 줄무늬", fontsize=12); axes[1].set_xlabel("측선 번호")
alt = 40 + 15 * np.sin(2 * np.pi * t / 120) + 5 * rng.normal(size=t.size).cumsum() / 40
axes[2].plot(t, alt, color="#1f4e79"); axes[2].set_title("지형 추종 비행의 고도 변화 → 진폭 변화", fontsize=12); axes[2].set_xlabel("초"); axes[2].set_ylabel("지상고 (m)")
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("드론 자력탐사 고유의 잡음원 — 유인 항공용 처리 흐름에는 없는 단계가 필요하다", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "concept_drone_noise.png", dpi=DPI); plt.close(fig)
print("concept figures done")
