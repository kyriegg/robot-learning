"""
题2: PD 控制 - 三种阻尼情况对比
对应作业: m·q̈(t) = u(t),  u(t) = -kp·q(t) - kd·q̇(t)

作业关键结论(b 题):
  临界阻尼条件: kp = m/τ² , kd = 2m/τ
  此时系统 q(t) 单调收敛到 0,无振荡

我们要做四件事:
  1) 数值模拟微分方程(简单 Euler 积分)
  2) 画三种阻尼的对比图
  3) 用解析解(作业公式 8-11)叠加验证
  4) 演示 τ 的物理意义(改变 τ 怎么影响响应速度)
"""
import numpy as np
import matplotlib
# 在 WSL 里如果没装 GUI,用 'Agg' 后端;有 GUI 就用 'TkAgg'
# 先尝试默认后端,失败就切到 Agg
try:
    import matplotlib.pyplot as plt
    plt.figure()  # 测试能不能开窗
    plt.close()
except Exception:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


# ==================== 1. 系统参数 ====================
m   = 1.0      # 质量 [kg]
tau = 0.5      # 期望时间常数 [s]: 系统在 τ 秒内收敛到约 1/e ≈ 37%
q0  = 1.0      # 初始位置 q(0)
T   = 3.0      # 仿真总时长 [s]
dt  = 0.001    # 时间步长 [s] (越小越精确)

# 临界阻尼增益 (作业 b 题答案)
kp = m / tau**2
kd_critical = 2 * m / tau

print("="*60)
print(f"  系统参数: m={m} kg,  τ={tau} s")
print(f"  作业公式 b: kp = m/τ² = {kp:.2f}")
print(f"             kd = 2m/τ = {kd_critical:.2f} (临界阻尼)")
print("="*60)


# ==================== 2. 数值模拟函数 ====================
def simulate(kp, kd, m=1.0, q0=1.0, T=3.0, dt=0.001):
    """
    用半隐式 Euler 法积分:
        m·q̈ = -kp·q - kd·q̇
    返回: 时间数组 t, 位置数组 q
    """
    n = int(T/dt)
    q, qdot = q0, 0.0
    history = np.empty(n)
    for i in range(n):
        u     = -kp*q - kd*qdot          # PD 控制律
        qddot = u/m                       # 牛顿第二定律
        qdot += qddot*dt                  # 速度更新
        q    += qdot*dt                   # 位置更新
        history[i] = q
    t = np.linspace(0, T, n)
    return t, history


# ==================== 3. 主对比图:三种阻尼 ====================
# 阻尼比 ξ 定义为 kd / (2 m / τ) = kd / kd_critical
xis = [0.2, 0.5, 1.0, 2.0]
labels = ['underdamped', 'underdamped', 'critical', 'overdamped']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig1, ax1 = plt.subplots(figsize=(10, 6))

for xi, lab, c in zip(xis, labels, colors):
    kd = xi * kd_critical
    t, q = simulate(kp, kd, m, q0, T, dt)
    ax1.plot(t, q, color=c, linewidth=2,
             label=f'ξ={xi}  ({lab}),  kd={kd:.2f}')

# 临界阻尼的解析解 (重根情况):
#   q(t) = (1 + t/τ)·e^(-t/τ)
# 这是从作业公式 (8) 当判别式 = 0 时推出来的
t_an = np.linspace(0, T, 1000)
q_analytic_critical = (1 + t_an/tau) * np.exp(-t_an/tau)
ax1.plot(t_an, q_analytic_critical, '--k', linewidth=1, alpha=0.6,
         label='analytic ξ=1: (1+t/τ)·e^(-t/τ)')

ax1.axhline(0, color='gray', linewidth=0.5)
ax1.set_xlabel('time [s]')
ax1.set_ylabel('q(t)')
ax1.set_title(f'PD control: damping comparison  (m={m}, kp={kp}, τ={tau}s)')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pd_damping_comparison.png', dpi=120)
print("\n[1/2] 已保存: pd_damping_comparison.png")


# ==================== 4. 第二张图: τ 的物理意义 ====================
# 作业 b 题告诉我们 τ 是"时间常数"
# 不同 τ 对应不同响应速度 — 都是临界阻尼
fig2, ax2 = plt.subplots(figsize=(10, 6))

taus = [0.2, 0.5, 1.0]
for tau_i, c in zip(taus, ['tab:purple', 'tab:green', 'tab:cyan']):
    kp_i = m / tau_i**2
    kd_i = 2 * m / tau_i
    t, q = simulate(kp_i, kd_i, m, q0, T, dt)
    ax2.plot(t, q, color=c, linewidth=2,
             label=f'τ={tau_i}s  →  kp={kp_i:.0f}, kd={kd_i:.0f}')

ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axhline(q0/np.e, color='red', linestyle=':', linewidth=1,
            label=f'1/e level (~ q reaches this at t = tau)')
ax2.set_xlabel('time [s]')
ax2.set_ylabel('q(t)')
ax2.set_title('Critical damping with different τ — τ controls response speed')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pd_tau_effect.png', dpi=120)
print("[2/2] 已保存: pd_tau_effect.png")


# ==================== 5. 终端输出关键数值 ====================
print("\n" + "="*60)
print("           关键时间点的数值验证")
print("="*60)

# 临界阻尼下,t=τ 时 q 应该是多少?
# 解析解: q(τ) = (1+1)·e^(-1) = 2/e ≈ 0.7358
t_, q_ = simulate(kp, kd_critical, m, q0, T, dt)
idx_tau = int(tau/dt)
print(f"\nξ=1 (临界), 在 t=τ={tau}s 时:")
print(f"  数值模拟  q(τ) = {q_[idx_tau]:.4f}")
print(f"  解析解    q(τ) = (1+1)·e⁻¹ = 2/e = {2/np.e:.4f}")
print(f"  匹配!" if abs(q_[idx_tau] - 2/np.e) < 0.01 else "  ⚠️  有偏差")

# 欠阻尼下,过冲量
t_, q_ = simulate(kp, 0.2*kd_critical, m, q0, T, dt)
overshoot = abs(q_.min())
print(f"\nξ=0.2 (强欠阻尼) 过冲量(下冲深度): {overshoot:.4f}")
print(f"  → 振荡的物理直觉: 阻尼小,系统'冲过头'再回来")

print("\n" + "="*60)
print("结论(对应作业 b 题):")
print("  临界阻尼 ξ=1 给出最快的'无振荡'收敛")
print("  小 ξ → 振荡 (机械臂会甩来甩去)")
print("  大 ξ → 反应迟钝 (机械臂慢吞吞)")
print("  这就是工业机器人调参的核心原则")
print("="*60)

# 如果有 GUI 后端,弹出窗口
try:
    plt.show()
except Exception:
    pass