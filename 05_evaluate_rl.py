"""
05_evaluate_rl.py (V2)
对比 LQR vs PPO 在不同参数随机化强度下的表现
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from stable_baselines3 import PPO

from segway_env import SegwayEnv


def design_lqr_for_params(params):
    """基于给定参数设计 LQR"""
    m_A, m_B = params['m_A'], params['m_B']
    I_A, I_B = params['I_A'], params['I_B']
    r, l = params['r'], params['l']
    g = 9.81
    
    a = m_A + m_B + I_A/r**2
    b = m_B * l
    c = m_B * l**2 + I_B
    M_inv = np.linalg.inv(np.array([[a, b], [b, c]]))
    
    A = np.zeros((4, 4))
    A[0, 2] = 1; A[1, 3] = 1
    A[2, 1] = M_inv[0, 1] * m_B * l * g
    A[3, 1] = M_inv[1, 1] * m_B * l * g
    B = np.zeros((4, 1))
    B[2, 0] = M_inv[0, 0]
    B[3, 0] = M_inv[1, 0]
    
    Q = np.diag([1.0, 100.0, 1.0, 10.0])
    R = np.array([[0.1]])
    P = solve_continuous_are(A, B, Q, R)
    return (np.linalg.inv(R) @ B.T @ P).flatten()


# 标称参数下的 LQR (这就是 LQR 的局限: 不知道运行时真实参数)
K_nominal = design_lqr_for_params(SegwayEnv.DEFAULT_PARAMS)
print(f"LQR 增益 (基于标称参数): K = {K_nominal}")

# 加载 PPO
print("加载 PPO 模型...")
try:
    model = PPO.load("segway_models/best_model")
    print("  使用 best_model.zip")
except FileNotFoundError:
    model = PPO.load("segway_ppo_final")
    print("  使用 segway_ppo_final.zip")


def evaluate_controller(controller_fn, randomize_strength, n_trials=50):
    env = SegwayEnv(randomize=(randomize_strength > 0),
                    randomize_strength=randomize_strength)
    rewards = []
    fails = 0
    for trial in range(n_trials):
        obs, _ = env.reset(seed=10000 + trial)
        total_r = 0
        for step in range(500):
            action = controller_fn(obs)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            if term:
                fails += 1
                break
            if trunc:
                break
        rewards.append(total_r)
    return np.mean(rewards), fails / n_trials


def lqr_controller(obs):
    return np.array([float(np.clip(-K_nominal @ obs, -50, 50))], dtype=np.float32)


def ppo_controller(obs):
    action, _ = model.predict(obs, deterministic=True)
    return action


strengths = [0.0, 0.5, 1.0, 1.5, 2.0]

print("\n" + "="*70)
print("                LQR vs PPO 对比 (各 50 次试验)")
print("="*70)
print(f"{'强度':>6} {'LQR reward':>12} {'LQR fail':>10} {'PPO reward':>12} {'PPO fail':>10}")
print("-"*70)

results_lqr, results_ppo = [], []
for s in strengths:
    lqr_r, lqr_f = evaluate_controller(lqr_controller, s, n_trials=50)
    ppo_r, ppo_f = evaluate_controller(ppo_controller, s, n_trials=50)
    print(f"{s:>6.1f} {lqr_r:>12.1f} {lqr_f*100:>9.0f}% {ppo_r:>12.1f} {ppo_f*100:>9.0f}%")
    results_lqr.append((lqr_r, lqr_f))
    results_ppo.append((ppo_r, ppo_f))

# 画图
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(strengths, [r[0] for r in results_lqr], 'r-o', linewidth=2, label='LQR')
axes[0].plot(strengths, [r[0] for r in results_ppo], 'g-s', linewidth=2, label='PPO')
axes[0].set_xlabel('Domain randomization strength')
axes[0].set_ylabel('Average cumulative reward')
axes[0].set_title('Performance vs. parameter uncertainty')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(strengths, [r[1]*100 for r in results_lqr], 'r-o', linewidth=2, label='LQR')
axes[1].plot(strengths, [r[1]*100 for r in results_ppo], 'g-s', linewidth=2, label='PPO')
axes[1].set_xlabel('Domain randomization strength')
axes[1].set_ylabel('Failure rate [%]')
axes[1].set_title('Failure rate (pendulum falls)')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lqr_vs_ppo_comparison.png', dpi=120)
print("\n已保存对比图: lqr_vs_ppo_comparison.png")

# 关键结论
lqr_2 = results_lqr[-1]
ppo_2 = results_ppo[-1]
print("\n" + "="*70)
print("                      关键结论")
print("="*70)
print(f"\n在强度 2.0 (±100% 参数变化) 下:")
print(f"  LQR 失败率: {lqr_2[1]*100:.0f}%, 平均 reward: {lqr_2[0]:.1f}")
print(f"  PPO 失败率: {ppo_2[1]*100:.0f}%, 平均 reward: {ppo_2[0]:.1f}")

if ppo_2[1] < lqr_2[1] - 0.05:
    print("\n✓ PPO 更鲁棒! 这就是 RL 在面对模型不确定性时的价值")
elif abs(ppo_2[1] - lqr_2[1]) < 0.05:
    print("\n○ PPO 与 LQR 表现相近 (训练目标和 LQR 几乎一致)")
else:
    print(f"\n⚠ PPO 还需要更多训练 (当前 reward {ppo_2[0]:.0f} < LQR {lqr_2[0]:.0f})")
    print("  建议: 重新跑 04_train_segway_rl.py 增加 TOTAL_TIMESTEPS 到 1_000_000")
print("="*70)
