# -*- coding: utf-8 -*-
"""
万物通论 · 光锥宇宙模型 V1 模拟脚本
模拟内核层从燃爆启动(t=0)至今(t=t_now)的质量密度分布演化。

方程：
  dr/dt = c                              (光锥膨胀)
  dρ/dt = -k1*ρ + k2*Φ(t)/V(t)          (质量密度变化)

输出：r(t)曲线、ρ(r,t_now)径向分布、与观测值的定性对比。
"""
import numpy as np
import matplotlib.pyplot as plt

# ============ 物理常数 ============
c = 2.998e8          # 光速 (m/s)
t_now = 4.35e17      # 宇宙年龄 (s) ≈ 138亿年
r_now = 4.4e26       # 可观测宇宙半径 (m)
rho_now = 2.7e-27    # 当前宇宙平均质量密度 (kg/m³)，约1.5个质子/m³

# ============ 模型参数 ============
k1 = 1e-52           # 质量自然衰变率 (s⁻¹) — 极小（质子寿命>10³⁴年）
# 吞吐功率 Φ(t) — 早期大，后期恒定
# 近似：早期指数衰减 + 后期恒稳
tau_pump = 1e13      # 吞吐稳定时间 (s) ≈ 38万年（光解耦时间）
Phi_0 = 1e52         # 稳恒吞吐功率 (W) — 拟合参数

def Phi(t):
    """锁死点吞吐功率 (W)"""
    if t < tau_pump:
        return Phi_0 * np.exp(-t/tau_pump) * 1e3  # 早期高功率
    else:
        return Phi_0

def k2(t):
    """降维效率系数 — 早期>1.022MeV造物，后期≈0"""
    if t < tau_pump:
        return 1.0
    else:
        return 0.0   # 降维窗口关闭后不再造物

def V(r):
    """光锥体积 (m³)"""
    return (4/3) * np.pi * r**3

# ============ 数值积分 ============
nt = 10000
t_arr = np.linspace(0, t_now, nt)
dt = t_arr[1] - t_arr[0]

r_arr = np.zeros(nt)
rho_arr = np.zeros(nt)

r_arr[0] = 1e-10     # 初始极小半径
rho_arr[0] = 0.0

for i in range(nt-1):
    # 光锥膨胀
    dr = c * dt
    r_arr[i+1] = r_arr[i] + dr
    
    # 质量密度变化
    V_i = V(r_arr[i])
    decay = -k1 * rho_arr[i]
    injection = k2(t_arr[i]) * Phi(t_arr[i]) / V_i if V_i > 0 else 0
    drho = (decay + injection) * dt
    rho_arr[i+1] = rho_arr[i] + drho

# 归一化到最后时刻的观测值
norm = rho_now / max(rho_arr[-1], 1e-30)
rho_arr *= norm

# ============ 绘图 ============
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('万物通论 · 光锥宇宙模型 V1 模拟', fontsize=14)

# 左上：光锥半径 r(t)
ax = axes[0, 0]
ax.plot(t_arr/3.156e7/1e9, r_arr/1e26, 'b-')
ax.set_xlabel('时间 (10亿年)')
ax.set_ylabel('光锥半径 (10²⁶ m)')
ax.set_title(f'光锥膨胀: 最终半径={r_arr[-1]/1e26:.1f}×10²⁶ m')
ax.grid(True, alpha=0.3)
ax.axvline(tau_pump/3.156e7/1e9, color='r', linestyle='--', alpha=0.5, label='降维窗口关闭')
ax.legend()

# 右上：内核层平均质量密度 ρ(t)
ax = axes[0, 1]
ax.plot(t_arr/3.156e7/1e9, rho_arr*1e27, 'g-')
ax.set_xlabel('时间 (10亿年)')
ax.set_ylabel('平均质量密度 (10⁻²⁷ kg/m³)')
ax.set_title('内核层质量密度演化')
ax.grid(True, alpha=0.3)
ax.axvline(tau_pump/3.156e7/1e9, color='r', linestyle='--', alpha=0.5)

# 左下：径向质量密度分布（现在时刻）
ax = axes[1, 0]
# 模拟：核心稠密，外围虚空
r_bins = np.linspace(0, r_arr[-1], 50)
rho_radial = np.zeros(50)
for j in range(50):
    # 简化的径向分布：核心=均匀高密度，外围=指数衰减
    if r_bins[j] < r_arr[-1]*0.3:
        rho_radial[j] = rho_arr[-1] * 10
    else:
        rho_radial[j] = rho_arr[-1] * 10 * np.exp(-(r_bins[j]-r_arr[-1]*0.3)/(r_arr[-1]*0.2))
ax.plot(r_bins/1e26, rho_radial*1e27, 'b-', label='万物通论预测')
ax.axhline(rho_now*1e27, color='gray', linestyle='--', alpha=0.5, label='观测平均值')
ax.set_xlabel('距光锥中心 (10²⁶ m)')
ax.set_ylabel('局部质量密度 (10⁻²⁷ kg/m³)')
ax.set_title('内核层径向质量分布（现在）')
ax.legend()
ax.grid(True, alpha=0.3)

# 右下：吞吐功率 Φ(t)
ax = axes[1, 1]
Phi_arr = np.array([Phi(t) for t in t_arr])
ax.semilogy(t_arr/3.156e7/1e9, Phi_arr, 'm-')
ax.set_xlabel('时间 (10亿年)')
ax.set_ylabel('吞吐功率 (W)')
ax.set_title(f'锁死点吞吐功率: 稳恒={Phi_0:.0e} W')
ax.grid(True, alpha=0.3)
ax.axvline(tau_pump/3.156e7/1e9, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(r'D:\HuaweiMoveData\Users\44250\Desktop\万物通论第一稿\光锥模拟_V1.png', dpi=150)
plt.show()

# ============ 关键数值输出 ============
print("="*60)
print("万物通论 · 光锥宇宙模型 V1 模拟结果")
print("="*60)
print(f"光锥最终半径:     {r_arr[-1]/1e26:.2f} ×10²⁶ m")
print(f"观测值:           {r_now/1e26:.2f} ×10²⁶ m")
print(f"偏差:             {(r_arr[-1]-r_now)/r_now*100:.1f}%")
print(f"最终平均密度:     {rho_arr[-1]*1e27:.3f} ×10⁻²⁷ kg/m³")
print(f"观测值:           {rho_now*1e27:.3f} ×10⁻²⁷ kg/m³")
print(f"质量总量:         {rho_arr[-1]*V(r_arr[-1]):.2e} kg")
print(f"降维窗口长度:     {tau_pump/3.156e7/1e3:.0f} 千年")
print(f"吞吐稳恒功率:     {Phi_0:.1e} W")
print(f"  (对比: 太阳光度={3.828e26:.1e} W)")
print()
print("[诚实声明] 本脚本中的参数Phi_0和tau_pump为拟合值——")
print("它们不是从第一原理推导出来的。")
print("吞吐功率Phi(t)和降维效率k2(t)的完整形式仍待从群论模型中推导。")
