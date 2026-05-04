"""
题3: 倒立摆 Segway 动力学仿真 + 平衡控制
对应作业: Euler-Lagrange 方程推导 (公式 26, 28)

设计:
  L1) 自由演化: 不施加控制 → 摆杆自己倒下来(验证不稳定动力学)
  L2) LQR 平衡: 加状态反馈 → 摆杆从倾斜状态自己站起来不倒
  L3) 位置跟踪: 让 Segway 走到目标 x → 观察"先后倾再前倾"的物理本能

依赖: numpy, scipy, matplotlib
  pip install numpy scipy matplotlib
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are


# ==================== 1. 系统参数 ====================
m_A, m_B = 1.0, 0.5     # 轮子, 摆杆质量 [kg]
I_A, I_B = 0.01, 0.05   # 转动惯量 [kg·m²]
r, l     = 0.1, 0.5     # 轮半径, 摆杆长度(到 COM)[m]
g        = 9.81

print("="*60)
print("  Segway 系统参数:")
print(f"    轮: m_A={m_A}kg, I_A={I_A}, r={r}m")
print(f"    摆: m_B={m_B}kg, I_B={I_B}, l={l}m")
print("="*60)


# ==================== 2. 动力学方程 (作业 26, 28) ====================
def dynamics(state, u):
    """
    state = [x, theta, x_dot, theta_dot]
    u     = 施加在轮子上的水平力 (对应作业 tau_1)

    作业方程整理为 M(q)·q̈ = RHS(q, q̇, u):
      M = [[m_A + m_B + I_A/r²,   m_B·cos(θ)·l    ],
           [m_B·cos(θ)·l,          m_B·l² + I_B    ]]
      RHS = [u + m_B·θ̇²·sin(θ)·l,
             m_B·l·sin(θ)·g      ]   (摆杆欠驱动: tau_2 = 0)

    返回: d/dt[x, theta, x_dot, theta_dot]
    """
    x, th, xd, thd = state
    c, s = np.cos(th), np.sin(th)

    M = np.array([
        [m_A + m_B + I_A/r**2,   m_B*c*l         ],
        [m_B*c*l,                 m_B*l**2 + I_B  ]
    ])
    rhs = np.array([
        u + m_B*thd**2*s*l,
        m_B*l*s*g
    ])
    qdd = np.linalg.solve(M, rhs)  # [ẍ; θ̈]
    return np.array([xd, thd, qdd[0], qdd[1]])


def rk4_step(state, u, dt):
    """4 阶 Runge-Kutta 积分 (比 Euler 精确得多)"""
    k1 = dynamics(state, u)
    k2 = dynamics(state + 0.5*dt*k1, u)
    k3 = dynamics(state + 0.5*dt*k2, u)
    k4 = dynamics(state + dt*k3, u)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6


# ==================== 3. LQR 控制器设计 ====================
# 在 theta=0 处线性化作业方程,然后求解 Riccati 方程得 K
def design_lqr():
    a = m_A + m_B + I_A/r**2
    b_ = m_B * l
    c_ = m_B * l**2 + I_B
    M_lin = np.array([[a, b_], [b_, c_]])
    M_inv = np.linalg.inv(M_lin)

    # 线性化: dot(state) = A·state + B·u
    A = np.zeros((4, 4))
    A[0, 2] = 1
    A[1, 3] = 1
    A[2, 1] = M_inv[0, 1] * m_B * l * g
    A[3, 1] = M_inv[1, 1] * m_B * l * g
    B = np.zeros((4, 1))
    B[2, 0] = M_inv[0, 0]
    B[3, 0] = M_inv[1, 0]

    # 代价权重: 强惩罚 theta 偏差(摔倒最严重),x 适中,速度更轻
    Q = np.diag([1.0, 100.0, 1.0, 10.0])
    R = np.array([[0.1]])
    P = solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P).flatten()
    return K

K = design_lqr()
print(f"  LQR 增益 K = [{K[0]:+.2f}, {K[1]:+.2f}, {K[2]:+.2f}, {K[3]:+.2f}]")
print(f"             [   x,    theta,  x_dot, theta_dot]")
print()


# ==================== 4. 三个场景的仿真 ====================
dt = 0.01
T  = 8.0
n_steps = int(T/dt)
t_arr = np.arange(n_steps) * dt


def simulate(scenario):
    """
    scenario:
      'L1_freefall'  - 无控制,看摆杆倒下
      'L2_balance'   - LQR,从倾斜状态站起
      'L3_tracking'  - LQR + 位置目标,走到 x=1
    """
    if scenario == 'L1_freefall':
        state = np.array([0.0, 0.05, 0.0, 0.0])
        target = np.zeros(4)
        u_func = lambda s: 0.0  # 没有控制
    elif scenario == 'L2_balance':
        state = np.array([0.0, 0.25, 0.0, 0.0])  # 倾斜 14°
        target = np.zeros(4)
        u_func = lambda s: float(np.clip(-K @ s, -50, 50))
    elif scenario == 'L3_tracking':
        state = np.array([0.0, 0.0, 0.0, 0.0])
        target = np.array([1.0, 0.0, 0.0, 0.0])
        u_func = lambda s: float(np.clip(-K @ (s - target), -100, 100))

    history = np.zeros((n_steps, 4))
    u_hist = np.zeros(n_steps)
    for i in range(n_steps):
        u = u_func(state)
        state = rk4_step(state, u, dt)
        history[i] = state
        u_hist[i] = u

    return history, u_hist, target


# ==================== 5. 可视化函数 ====================
def draw_segway(ax, x, theta, color='steelblue', alpha=1.0):
    """画一个 Segway: 轮子 + 摆杆 + 顶部质量块"""
    # 轮子
    wheel = Circle((x, r), radius=r, fill=False, edgecolor='black',
                   linewidth=2, alpha=alpha)
    ax.add_patch(wheel)
    # 轮辐(显示旋转): 滚动条件 phi = -x/r
    rot = -x / r
    rx1 = x + r * np.cos(rot); rz1 = r + r * np.sin(rot)
    rx2 = x - r * np.cos(rot); rz2 = r - r * np.sin(rot)
    ax.plot([rx1, rx2], [rz1, rz2], 'k-', linewidth=1, alpha=alpha)
    # 摆杆
    pole_top_x = x + l * np.sin(theta)
    pole_top_z = r + l * np.cos(theta)
    ax.plot([x, pole_top_x], [r, pole_top_z],
            color=color, linewidth=4, alpha=alpha, solid_capstyle='round')
    # 顶部质量块
    bs = 0.10
    block = Rectangle((-bs/2, -bs/2), bs, bs,
                      facecolor=color, edgecolor='black',
                      linewidth=1.5, alpha=alpha)
    block.set_transform(Affine2D().rotate(theta)
                        .translate(pole_top_x, pole_top_z) + ax.transData)
    ax.add_patch(block)


def make_animation(history, u_hist, target, title, save_path,
                   color='steelblue', frame_skip=4):
    """生成动画并保存为 mp4 / gif。
    
    frame_skip=4 在 dt=0.01 下意味着 25 fps 的动画,8 秒视频共 200 帧。
    保存大约需 5-15 秒(取决于电脑速度)。
    """
    fig, (ax_anim, ax_traj) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}
    )

    # 上半部分: Segway 动画
    x_min = min(history[:, 0].min(), 0) - 0.4
    x_max = max(history[:, 0].max(), 0) + 0.4
    ax_anim.set_xlim(x_min, x_max)
    ax_anim.set_ylim(-0.15, 0.85)
    ax_anim.set_aspect('equal')
    ax_anim.axhline(0, color='saddlebrown', linewidth=2)
    if target[0] != 0:
        ax_anim.axvline(target[0], color='red', linestyle='--',
                        alpha=0.5, label=f'target x = {target[0]}')
        ax_anim.legend(loc='upper right')
    ax_anim.set_title(title, fontsize=13)
    ax_anim.grid(alpha=0.3)
    ax_anim.set_xlabel('x [m]')

    # 下半部分: theta 轨迹
    ax_traj.plot(t_arr, np.degrees(history[:, 1]), 'b-',
                 linewidth=2, label='theta [deg]')
    ax_traj.axhline(0, color='gray', linewidth=0.5)
    ax_traj.set_xlabel('time [s]')
    ax_traj.set_ylabel('theta [deg]')
    ax_traj.legend(loc='upper right')
    ax_traj.grid(alpha=0.3)
    cursor = ax_traj.axvline(0, color='red', linewidth=1)

    def update(frame):
        idx = frame * frame_skip
        if idx >= n_steps:
            idx = n_steps - 1
        # 清掉上次画的 Segway
        for patch in list(ax_anim.patches):
            patch.remove()
        for line in list(ax_anim.lines):
            if line.get_linestyle() == '--':  # 保留参考线
                continue
            line.remove()
        ax_anim.axhline(0, color='saddlebrown', linewidth=2)

        x_now = history[idx, 0]
        th_now = history[idx, 1]
        draw_segway(ax_anim, x_now, th_now, color=color)

        # 状态文本
        info = (f"t={idx*dt:.2f}s  x={x_now:+.3f}m  "
                f"theta={np.degrees(th_now):+.2f}deg  u={u_hist[idx]:+.2f}N")
        ax_anim.set_title(f"{title}\n{info}", fontsize=11)

        # 时间游标
        cursor.set_xdata([idx*dt, idx*dt])
        return []

    n_frames = n_steps // frame_skip
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=40, blit=False)
    plt.tight_layout()

    # 保存
    try:
        # 优先 mp4 (需 ffmpeg)
        anim.save(save_path + '.mp4', fps=25, dpi=80,
                  extra_args=['-vcodec', 'libx264'])
        print(f"  已保存: {save_path}.mp4")
    except Exception:
        # 回退到 gif (不需 ffmpeg)
        try:
            anim.save(save_path + '.gif', fps=30, dpi=80, writer='pillow')
            print(f"  已保存: {save_path}.gif")
        except Exception as e:
            print(f"  动画保存失败: {e}")
            print(f"  改为保存关键帧静态图: {save_path}_frames.png")
            save_keyframes(history, target, title, save_path + '_frames.png',
                           color=color)
    plt.close(fig)
    return anim


def save_keyframes(history, target, title, save_path, color='steelblue',
                   n_keyframes=6):
    """如果动画保存不了,至少存几个关键帧的截图"""
    fig, axes = plt.subplots(1, n_keyframes, figsize=(3*n_keyframes, 4))
    indices = np.linspace(0, n_steps-1, n_keyframes).astype(int)
    x_min = history[:, 0].min() - 0.4
    x_max = history[:, 0].max() + 0.4
    for ax, idx in zip(axes, indices):
        ax.axhline(0, color='saddlebrown', linewidth=2)
        if target[0] != 0:
            ax.axvline(target[0], color='red', linestyle='--', alpha=0.5)
        draw_segway(ax, history[idx, 0], history[idx, 1], color=color)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.15, 0.85)
        ax.set_aspect('equal')
        ax.set_title(f"t={idx*dt:.2f}s\ntheta={np.degrees(history[idx,1]):+.1f}deg",
                     fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"  已保存关键帧: {save_path}")


# ==================== 6. 跑三个场景 ====================
print("\n=== L1: 自由演化(无控制)===")
hist1, u1, tgt1 = simulate('L1_freefall')
print(f"  初始 theta = {np.degrees(hist1[0,1]):.2f}deg "
      f"(只偏 2.9deg) -> {T}s 后 theta = {np.degrees(hist1[-1,1]):.1f}deg")
print(f"  → 摆杆完全失控,验证: 不加控制时直立平衡是不稳定的!")
make_animation(hist1, u1, tgt1, 'L1: Free fall (no control)',
               'segway_L1_freefall', color='tab:red')

print("\n=== L2: LQR 平衡控制 ===")
hist2, u2, tgt2 = simulate('L2_balance')
print(f"  初始 theta = {np.degrees(hist2[0,1]):.2f}deg "
      f"-> {T}s 后 theta = {np.degrees(hist2[-1,1]):.4f}deg")
print(f"  → LQR 把摆杆从 14deg 拉回到几乎 0deg 直立!")
make_animation(hist2, u2, tgt2, 'L2: LQR balance control',
               'segway_L2_balance', color='tab:green')

print("\n=== L3: 位置跟踪(走到 x=1m)===")
hist3, u3, tgt3 = simulate('L3_tracking')
print(f"  目标 x = 1.0m, 最终 x = {hist3[-1,0]:.4f}m, "
      f"最终 theta = {np.degrees(hist3[-1,1]):.2f}deg")
peak_back = np.degrees(hist3[:200, 1].max())
peak_fwd  = np.degrees(hist3[200:, 1].min())
print(f"  → 启动时先后倾 +{peak_back:.2f}deg 积蓄速度,然后前倾 {peak_fwd:.2f}deg 加速")
print(f"  → 这就是真实 Segway 的物理本能!")
make_animation(hist3, u3, tgt3,
               'L3: position tracking (target x=1m)',
               'segway_L3_tracking', color='tab:blue')


# ==================== 7. 汇总图 ====================
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(t_arr, np.degrees(hist1[:, 1]), 'r-',
             linewidth=2, label='L1: free fall (no control)')
axes[0].plot(t_arr, np.degrees(hist2[:, 1]), 'g-',
             linewidth=2, label='L2: LQR balance')
axes[0].plot(t_arr, np.degrees(hist3[:, 1]), 'b-',
             linewidth=2, label='L3: position tracking')
axes[0].axhline(0, color='gray', linewidth=0.5)
axes[0].set_ylabel('theta [deg]')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_title('Pendulum tilt angle theta(t) across scenarios')

axes[1].plot(t_arr, hist1[:, 0], 'r-', linewidth=2, label='L1')
axes[1].plot(t_arr, hist2[:, 0], 'g-', linewidth=2, label='L2')
axes[1].plot(t_arr, hist3[:, 0], 'b-', linewidth=2, label='L3')
axes[1].axhline(1.0, color='blue', linestyle='--', alpha=0.5,
                label='L3 target x=1')
axes[1].set_xlabel('time [s]')
axes[1].set_ylabel('x [m]')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_title('Wheel position x(t) across scenarios')

plt.tight_layout()
plt.savefig('segway_summary.png', dpi=120)
plt.close(fig)
print("\n已保存汇总图: segway_summary.png")

print("\n" + "="*60)
print("  任务完成! 文件清单:")
print("    - segway_L1_freefall.{mp4|gif}  : 摆杆倒下")
print("    - segway_L2_balance.{mp4|gif}    : 自动平衡")
print("    - segway_L3_tracking.{mp4|gif}   : 走到目标")
print("    - segway_summary.png              : 三场景对比图")
print("="*60)