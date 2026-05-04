"""
题2 进阶: 真实 Panda 机械臂上的 PD 控制
对应作业: m·q̈(t) = -kp·(q - q*) - kd·q̇(t)

设计:
  左 Panda  - 欠阻尼 (ξ=0.3): 关节会"甩过头"再回来
  右 Panda  - 临界阻尼 (ξ=1.0): 平滑单调收敛

我们仅控制 joint4 (肘部) - 因为它运动幅度最大,效果最直观
其他关节锁住,这样可以专注观察 PD 的物理行为
"""
import robotic as ry
import numpy as np
import time


# ==================== 1. 搭建场景:两个并排的 Panda ====================
C = ry.Config()
C.addFile(ry.raiPath('panda/panda.g'), namePrefix='L_')
C.addFile(ry.raiPath('panda/panda.g'), namePrefix='R_')

C.getFrame('L_panda_base').setPosition([0, -0.6, 0])
C.getFrame('R_panda_base').setPosition([0,  0.6, 0])

# 在每个 Panda 的目标位置放一个小标记球(目标姿态时夹爪应该在的位置)
# Panda 默认 q=0 时夹爪大约在 [0.5, 0, 0.6] (相对基座)
C.addFrame('L_target_marker') \
    .setShape(ry.ST.sphere, [0.04]) \
    .setColor([0.2, 1.0, 0.2]) \
    .setPosition([0.5, -0.6, 1.0])
C.addFrame('R_target_marker') \
    .setShape(ry.ST.sphere, [0.04]) \
    .setColor([0.2, 1.0, 0.2]) \
    .setPosition([0.5,  0.6, 1.0])

# 取出当前关节状态。共 14 个: [L_joint1..7, R_joint1..7]
# Panda 默认 panda.g 已经把 joint4 设成 -2.0(肘部弯曲)
# 这正是我们想要的"初始偏离状态"
q_init = C.getJointState()
print(f"初始关节状态(每个 Panda 7 关节): {q_init}")

# 给两个 Panda 都设定相同的初始扰动
# joint4 = -2.0 (这就是 panda.g 的默认值,正好是个好起点)
q_state = q_init.copy()

# 关节速度(初始为 0)
qdot_state = np.zeros(14)


# ==================== 2. PD 控制器参数 ====================
# 我们假定每个关节有"虚拟质量" m,这是简化模型
# (真实机械臂每个关节看到的等效质量取决于其他关节配置,
#  但一阶近似下可以视为常数)
m_eff = 1.0    # 虚拟有效惯量
tau   = 0.5    # 期望时间常数 [s]

# 作业 b 题: 临界阻尼 kp = m/τ², kd = 2m/τ
kp = m_eff / tau**2          # = 4.0
kd_critical = 2 * m_eff / tau  # = 4.0

# 左 Panda:欠阻尼(ξ=0.3),会振荡
xi_L = 0.3
kd_L = xi_L * kd_critical    # = 1.2

# 右 Panda:临界阻尼(ξ=1.0)
xi_R = 1.0
kd_R = xi_R * kd_critical    # = 4.0

# 目标姿态:把 joint4 从 -2.0 推回到 0(肘部伸直)
JOINT_INDEX = 3  # Panda 7 关节里的 joint4 (索引从 0 开始)
q_target_local = 0.0  # 目标值

print(f"\nPD 参数:")
print(f"  虚拟质量 m={m_eff}, 时间常数 τ={tau}s")
print(f"  kp = {kp}")
print(f"  左 Panda(欠阻尼 ξ={xi_L}): kd = {kd_L}")
print(f"  右 Panda(临界  ξ={xi_R}): kd = {kd_R}")
print(f"  目标: joint4 从 -2.0 → {q_target_local}")


# ==================== 3. 显示初始状态 ====================
C.view(True, "初始状态: 两个 Panda 都把 joint4 弯到 -2.0 - 按键开始 PD 控制")


# ==================== 4. 仿真循环 ====================
dt = 0.01           # 物理仿真步长 (越小越精确)
T = 4.0             # 仿真总时长
view_dt = 0.02      # 视图更新间隔 (太快眼睛跟不上)
n_steps = int(T/dt)
view_every = int(view_dt/dt)

# 记录轨迹用于后面分析
history_t = []
history_qL = []
history_qR = []

t_sim = 0.0
last_view_time = time.time()

for step in range(n_steps):
    # ---- 左 Panda 的 PD 控制 ----
    qL_idx = JOINT_INDEX                              # joint4 在向量中的位置 = 3
    err_L = q_state[qL_idx] - q_target_local
    u_L = -kp * err_L - kd_L * qdot_state[qL_idx]
    qddot_L = u_L / m_eff
    qdot_state[qL_idx] += qddot_L * dt
    q_state[qL_idx]    += qdot_state[qL_idx] * dt

    # ---- 右 Panda 的 PD 控制 ----
    qR_idx = 7 + JOINT_INDEX                          # joint4 在右 Panda = 7+3 = 10
    err_R = q_state[qR_idx] - q_target_local
    u_R = -kp * err_R - kd_R * qdot_state[qR_idx]
    qddot_R = u_R / m_eff
    qdot_state[qR_idx] += qddot_R * dt
    q_state[qR_idx]    += qdot_state[qR_idx] * dt

    # ---- 记录 ----
    history_t.append(t_sim)
    history_qL.append(q_state[qL_idx])
    history_qR.append(q_state[qR_idx])

    # ---- 更新视图(不是每步都更新,避免太密)----
    if step % view_every == 0:
        C.setJointState(q_state)
        C.view(False,
               f"t={t_sim:.2f}s  |  L(欠阻尼) q4={q_state[qL_idx]:+.3f}"
               f"  |  R(临界) q4={q_state[qR_idx]:+.3f}")
        # 实时同步:让仿真时间和真实时间近似匹配
        elapsed = time.time() - last_view_time
        if elapsed < view_dt:
            time.sleep(view_dt - elapsed)
        last_view_time = time.time()

    t_sim += dt


# ==================== 5. 仿真结束:数值统计 ====================
history_t  = np.array(history_t)
history_qL = np.array(history_qL)
history_qR = np.array(history_qR)

# 过冲量:欠阻尼会越过 0 到正值
overshoot_L = max(0, history_qL.max())  # 越过 0 的最大正值
overshoot_R = max(0, history_qR.max())

# 第一次"基本到位"(在 ±0.1 内)的时间
def settling_time(q, target, tol=0.1):
    for i in range(len(q)-1, -1, -1):
        if abs(q[i] - target) > tol:
            return history_t[min(i+1, len(history_t)-1)]
    return 0.0

settle_L = settling_time(history_qL, q_target_local)
settle_R = settling_time(history_qR, q_target_local)

print("\n" + "="*60)
print("                   仿真结果")
print("="*60)
print(f"{'指标':<24} {'左(欠阻尼)':<18} {'右(临界)':<18}")
print("-"*60)
print(f"{'最终 q4':<24} {history_qL[-1]:<+18.4f} {history_qR[-1]:<+18.4f}")
print(f"{'过冲量(越过0)':<24} {overshoot_L:<18.4f} {overshoot_R:<18.4f}")
print(f"{'稳定时间(到±0.1)':<24} {settle_L:<18.2f} {settle_R:<18.2f}")
print("="*60)
print()
print("解读:")
print(f"  左 Panda(ξ={xi_L}): 振荡 + 过冲,但震荡平息后到位")
print(f"  右 Panda(ξ={xi_R}):  平滑单调收敛,不过冲")
print()
print("这就是工业机器人为什么调到临界阻尼附近 ξ ≈ 0.7-1.0:")
print("  - 太小 → 关节抖动,机械寿命下降")
print("  - 太大 → 反应迟钝,生产效率低")


# ==================== 6. 保存轨迹图 ====================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_t, history_qL, color='tab:blue', linewidth=2,
            label=f'Left Panda (xi={xi_L}, underdamped)')
    ax.plot(history_t, history_qR, color='tab:green', linewidth=2,
            label=f'Right Panda (xi={xi_R}, critical)')
    ax.axhline(q_target_local, color='red', linestyle='--', alpha=0.6,
               label=f'target q* = {q_target_local}')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('joint4 position [rad]')
    ax.set_title('PD control on real Panda joint4: under-damped vs critical')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('panda_pd_trajectory.png', dpi=120)
    print("\n已保存轨迹图: panda_pd_trajectory.png")
except ImportError:
    print("\n(matplotlib 未安装,跳过画图)")


# ==================== 7. 最终展示 ====================
C.view(True, "仿真结束 - 按键退出")