import sys
import numpy as np
import matplotlib.pyplot as plt
from batman import TransitParams, TransitModel
from numpy.random import normal

# === 常数 ===
G = 6.67430e-11           # 万有引力常数，单位：m^3 kg^-1 s^-2
M_sun = 1.98847e30        # 太阳质量，单位：kg
AU = 1.495978707e11       # 天文单位，单位：m
seconds_per_day = 86400   # 每天的秒数

# === 读取输入参数 ===
# 输入格式： a_AU, ecc, inc, M_star_solar, rp, noise
a_AU = float(sys.argv[1])         # 半长轴，单位：AU
ecc = float(sys.argv[2])          # 离心率
inc = float(sys.argv[3])          # 倾角，单位：度
M_star_solar = float(sys.argv[4]) # 恒星质量，单位：太阳质量
rp = float(sys.argv[5])           # 行星半径（单位：R_star 的比例）
noise = float(sys.argv[6])/1e6        # 噪声标准差（乘性，单位：无）

# === 计算周期（单位：天） ===
M_star = M_star_solar * M_sun     # 转换为 kg
a = a_AU * AU                     # 转换为 m
P_sec = 2 * np.pi * np.sqrt(a**3 / (G * M_star))  # 周期（秒）
per = P_sec / seconds_per_day    # 转换为天

# === 设置凌星参数 ===
params = TransitParams()
params.per = per
params.t0 = per / 2               # 中点对齐
params.rp = rp
params.a = a_AU                   # 仍使用 AU，batman 支持单位归一化
params.inc = inc
params.ecc = ecc
params.w = 90.0                   # 近地点角设为90°
params.limb_dark = "quadratic"
params.u = [0.1, 0.3]             # 默认二次暗边参数

# === 时间轴设置 ===
t = np.linspace(- per * 1.5, per * 1.5, 6000)
m = TransitModel(params, t)
flux = m.light_curve(params)

real_depth = (1 - np.min(flux) ) * 1e6

# === 添加噪声 ===
flux_noisy = flux * normal(1, noise, size=flux.shape)

# === 转换为 ppm 单位 ===
flux_ppm = (flux - 1) * 1e6
flux_noisy_ppm = (flux_noisy - 1) * 1e6


def detect_transit_durations_by_count(model_flux, cadence_min=29.4244):
    """
    根据 flux < 1 的连续段长度估算凌星持续时间（单位：天）

    参数:
    - model_flux: ndarray, 拟合光变数据
    - cadence_min: float, 每个采样点的时间间隔，默认 29.4 分钟（Kepler 长时采样）

    返回:
    - durations: list of float, 每次凌星的持续时间（单位：天）
    - lengths: list of int, 每次凌星包含的采样点数
    - indices: list of tuple, 每次凌星的起止下标 (start_idx, end_idx)
    """

    in_transit = model_flux < 1.0
    idx = np.where(in_transit)[0]
    if len(idx) == 0:
        return [], [], []

    # 找出连续段
    splits = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, splits + 1)

    durations = []
    lengths = []
    indices = []

    for segment in segments:
        n_points = len(segment)
        duration_days = (n_points * cadence_min) / (60 * 24)
        durations.append(duration_days)
        lengths.append(n_points)
        indices.append((segment[0], segment[-1]))

    return np.max(durations), lengths, indices

# === 估算持续时间（单位：天） ===
duration, _, _ = detect_transit_durations_by_count(flux, cadence_min=3*per*24*60/6000)

# === 绘图 ===
plt.figure(figsize=(6, 4))
plt.plot(t, flux_noisy_ppm, 'k.', label="Simulated (noisy)", markersize=1)
plt.plot(t, flux_ppm, 'r-', lw=1.5, label="Model")

# 添加对比线
plt.axhline(-30, color='gray', linestyle='--', label='30 ppm limit')
plt.axhline(-50, color='gray', linestyle=':', label='50 ppm limit')
plt.axhline(-real_depth, color='red', linestyle='--', label=f'Real: {int(real_depth)} ppm')

plt.xlabel("Time [days]")
plt.ylabel("Transit Depth [ppm]")
plt.title("Transit Light Curve")
plt.xlim(params.t0 - 1.8 * duration, params.t0 + 1.8 * duration)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("output_figure.png", dpi=600)
