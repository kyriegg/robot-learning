"""
04_train_segway_rl.py (V2)
用 PPO 训练 Segway,改进点:
  - 训练 500K 步 (V1 的 200K 不够)
  - 改用 V2 环境 (简化 reward)
  - 调整超参 (更适合鲁棒性训练)

预期训练时间:
  CPU: ~15-25 分钟
  GPU: ~5-10 分钟
"""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from segway_env import SegwayEnv


def make_env(randomize=True):
    def _f():
        return Monitor(SegwayEnv(randomize=randomize, randomize_strength=2.0))
    return _f


# 8 个并行环境 (越多越快,只要 CPU/RAM 跟得上)
train_env = make_vec_env(make_env(randomize=True), n_envs=8)
eval_env = Monitor(SegwayEnv(randomize=False))

print("="*60)
print("  PPO 训练 (V2 - 改进版)")
print("  - 8 个并行环境,500K 步")
print("  - 训练时随机化 m_B, l, I_B 各 ±50%")
print("="*60)

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,           # 比 V1 大,适合 8 个并行
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,            # 鼓励探索
    verbose=1,
    tensorboard_log="./segway_tb_log/",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./segway_models/",
    log_path="./segway_eval_log/",
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True,
)

TOTAL_TIMESTEPS = 500_000
print(f"\n训练 {TOTAL_TIMESTEPS} 步,大约 5-25 分钟...\n")

t_start = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    progress_bar=True,
)
t_train = time.time() - t_start
print(f"\n训练完成! 用时 {t_train/60:.1f} 分钟")

model.save("segway_ppo_final")
print("已保存: segway_ppo_final.zip 和 segway_models/best_model.zip")
print("\n下一步: python3 05_evaluate_rl.py")
