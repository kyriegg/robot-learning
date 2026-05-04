"""
Segway Gymnasium 环境 V3 - 真正适合 PPO 学习的版本
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SegwayEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    DEFAULT_PARAMS = {
        'm_A': 1.0, 'm_B': 0.5,
        'I_A': 0.01, 'I_B': 0.05,
        'r': 0.1, 'l': 0.5,
    }
    
    def __init__(self, randomize=False, randomize_strength=1.0):
        super().__init__()
        obs_high = np.array([5.0, np.pi/2, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-50.0, 50.0, shape=(1,), dtype=np.float32)
        
        self.randomize = randomize
        self.randomize_strength = randomize_strength
        self.dt = 0.02
        self.max_steps = 500
        self.g = 9.81
        self.steps = 0
        self.state = None
        self._set_default_params()
    
    def _set_default_params(self):
        for k, v in self.DEFAULT_PARAMS.items():
            setattr(self, k, v)
    
    def _randomize_params(self):
        s = self.randomize_strength
        self.m_B = max(0.1, self.DEFAULT_PARAMS['m_B'] * (1 + 0.5*s*self.np_random.uniform(-1, 1)))
        self.l   = max(0.15, self.DEFAULT_PARAMS['l']   * (1 + 0.5*s*self.np_random.uniform(-1, 1)))
        self.I_B = max(0.01, self.DEFAULT_PARAMS['I_B'] * (1 + 0.5*s*self.np_random.uniform(-1, 1)))
    
    def get_params(self):
        return {k: getattr(self, k) for k in self.DEFAULT_PARAMS}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize:
            self._randomize_params()
        else:
            self._set_default_params()
        # 课程学习: 训练时初始扰动较小,让 PPO 容易开局成功
        s = self.randomize_strength
        self.state = self.np_random.uniform(
            low =[-0.02, -0.035, -0.02, -0.05],
            high=[ 0.02,  0.035,  0.02,  0.05]
        ).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}
    
    def _dynamics(self, state, u):
        x, th, xd, thd = state
        c, s_ = np.cos(th), np.sin(th)
        M = np.array([
            [self.m_A + self.m_B + self.I_A/self.r**2, self.m_B*c*self.l],
            [self.m_B*c*self.l, self.m_B*self.l**2 + self.I_B]
        ])
        rhs = np.array([
            u + self.m_B*thd**2*s_*self.l,
            self.m_B*self.l*s_*self.g
        ])
        qdd = np.linalg.solve(M, rhs)
        return np.array([xd, thd, qdd[0], qdd[1]], dtype=np.float64)
    
    def step(self, action):
        u = float(np.clip(action[0], -50, 50))
        s_arr = self.state.astype(np.float64)
        k1 = self._dynamics(s_arr, u)
        k2 = self._dynamics(s_arr + 0.5*self.dt*k1, u)
        k3 = self._dynamics(s_arr + 0.5*self.dt*k2, u)
        k4 = self._dynamics(s_arr + self.dt*k3, u)
        self.state = (s_arr + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6).astype(np.float32)
        self.steps += 1
        
        x, th, xd, thd = self.state
        
        # ⭐ 关键改动: dense + 始终为正的 reward
        # 这样 PPO 即使在失败的 episode 中也能从前 N 步获得正向信号
        # 摔倒时不再给巨大负惩罚,只是 episode 早结束累计 reward 更少
        reward = (
            np.cos(th)                      # +1 直立时, +0 平躺时, 平滑奖励
            - 0.01 * th**2                  # 额外惩罚倾角(让奖励对小角度更敏感)
            - 0.0005 * u**2                 # 轻微惩罚动作
        )
        
        terminated = bool(abs(th) > np.pi/3 or abs(x) > 5.0)
        truncated = self.steps >= self.max_steps
        # 不再扣 -100 大惩罚
        
        return self.state.copy(), float(reward), terminated, truncated, {}