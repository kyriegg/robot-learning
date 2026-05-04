"""
diagnose.py
诊断为什么 PPO 在 ±100% 下还是输给 LQR

我们要回答 3 个问题:
1) PPO 失败的那 14%,是哪些参数组合?
2) LQR 失败的 4%,是哪些参数组合?
3) 失败的 episode 长什么样?
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from stable_baselines3 import PPO
from segway_env import SegwayEnv


# ==================== LQR 设计(同 05) ====================
def design_lqr(params):
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
    B[2, 0] = M_inv[0, 0]; B[3, 0] = M_inv[1, 0]
    Q = np.diag([1.0, 100.0, 1.0, 10.0])
    R = np.array([[0.1]])
    P = solve_continuous_are(A, B, Q, R)
    return (np.linalg.inv(R) @ B.T @ P).flatten()

K = design_lqr(SegwayEnv.DEFAULT_PARAMS)

# 加载 PPO
try:
    model = PPO.load("segway_models/best_model")
    print("使用 best_model.zip\n")
except FileNotFoundError:
    model = PPO.load("segway_ppo_final")
    print("使用 segway_ppo_final.zip\n")


# ==================== 诊断: 在强度 2.0 下,记录每次失败的细节 ====================
print("="*80)
print("  在强度 2.0 下,逐个分析 50 次 trial")
print("="*80)
print(f"{'#':>3} {'m_B':>6} {'l':>6} {'I_B':>7} | {'LQR':>10} {'PPO':>10} | {'verdict':>20}")
print("-"*80)

env = SegwayEnv(randomize=True, randomize_strength=2.0)

lqr_fail_params = []
ppo_fail_params = []
both_fail_params = []
ppo_only_fail_params = []
lqr_only_fail_params = []

for trial in range(50):
    # 重置环境(用相同 seed,确保 LQR 和 PPO 面对同样的参数和初始状态)
    obs_lqr, _ = env.reset(seed=10000 + trial)
    params = env.get_params()  # 拿这次的随机参数
    init_state = obs_lqr.copy()
    
    # ---- LQR ----
    lqr_steps = 0
    lqr_failed = False
    obs = obs_lqr.copy()
    for step in range(500):
        u = float(np.clip(-K @ obs, -50, 50))
        obs, r, term, trunc, _ = env.step(np.array([u], dtype=np.float32))
        lqr_steps += 1
        if term:
            lqr_failed = True
            break
        if trunc:
            break
    
    # ---- PPO (从同样的初始状态开始) ----
    obs_ppo, _ = env.reset(seed=10000 + trial)  # 同样的 seed -> 同样的参数和初值
    ppo_steps = 0
    ppo_failed = False
    obs = obs_ppo.copy()
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        ppo_steps += 1
        if term:
            ppo_failed = True
            break
        if trunc:
            break
    
    # 判断哪种失败
    if lqr_failed and ppo_failed:
        verdict = "都失败"
        both_fail_params.append((params, init_state))
    elif lqr_failed:
        verdict = "LQR失败,PPO赢"
        lqr_only_fail_params.append((params, init_state))
    elif ppo_failed:
        verdict = "PPO失败,LQR赢"
        ppo_only_fail_params.append((params, init_state))
    else:
        verdict = "都成功"
    
    # 只打印失败的
    if lqr_failed or ppo_failed:
        print(f"{trial:>3} {params['m_B']:>6.3f} {params['l']:>6.3f} {params['I_B']:>7.4f} | "
              f"{'X@'+str(lqr_steps) if lqr_failed else 'OK 500':>10} "
              f"{'X@'+str(ppo_steps) if ppo_failed else 'OK 500':>10} | {verdict:>20}")


# ==================== 汇总 ====================
print("\n" + "="*80)
print("                          诊断汇总")
print("="*80)
print(f"  LQR 仅失败: {len(lqr_only_fail_params)} 次")
print(f"  PPO 仅失败: {len(ppo_only_fail_params)} 次")
print(f"  双双失败:   {len(both_fail_params)} 次")
print(f"  双双成功:   {50 - len(lqr_only_fail_params) - len(ppo_only_fail_params) - len(both_fail_params)} 次")

# 看 PPO 失败的参数分布
if ppo_only_fail_params:
    print(f"\nPPO 单独失败的参数(共 {len(ppo_only_fail_params)} 次):")
    print(f"  {'m_B':>8} {'l':>8} {'I_B':>10}    {'init_theta':>12}")
    for params, state in ppo_only_fail_params:
        print(f"  {params['m_B']:>8.3f} {params['l']:>8.3f} {params['I_B']:>10.4f}    {np.degrees(state[1]):>+10.2f}°")
    
    # 找规律
    m_Bs = [p[0]['m_B'] for p in ppo_only_fail_params]
    ls = [p[0]['l'] for p in ppo_only_fail_params]
    print(f"\n  PPO 失败的 m_B 范围: [{min(m_Bs):.3f}, {max(m_Bs):.3f}]")
    print(f"  PPO 失败的 l 范围:   [{min(ls):.3f}, {max(ls):.3f}]")
    print(f"\n  对比训练分布: m_B ∈ [0.05, 1.0], l ∈ [0.05, 1.0]")
    print(f"  → 如果失败都集中在分布边缘,说明 PPO 在边缘还是没学好")