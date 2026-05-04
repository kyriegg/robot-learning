"""
题1: 逆运动学 - 双机械臂并排对比
对应作业公式 (1)(2)(3): min ||q-q₀||² + μ||φ(q)||²

左边 Panda: μ=0.5 (软约束) - 末端到不了目标
右边 Panda: μ→∞ (硬约束)  - 末端精确到达
"""
import robotic as ry
import numpy as np
import time

# ==================== 1. 搭建场景 ====================
C = ry.Config()

# 加载左、右两个 Panda(用 namePrefix 区分)
C.addFile(ry.raiPath('panda/panda.g'), namePrefix='L_')
C.addFile(ry.raiPath('panda/panda.g'), namePrefix='R_')

# 把两个基座往左右各偏移 0.6 米
C.getFrame('L_panda_base').setPosition([0, -0.6, 0])
C.getFrame('R_panda_base').setPosition([0,  0.6, 0])

# 设定目标(相对于各自基座的位置)
# 故意选一个机械臂需要"努力够"的位置,让 μ 的差异看得清
target_local = [0.6, 0.4, 0.3]

C.addFrame('L_target') \
    .setShape(ry.ST.sphere, [0.05]) \
    .setColor([1, 0.5, 0]) \
    .setPosition([target_local[0], -0.6 + target_local[1], target_local[2]])

C.addFrame('R_target') \
    .setShape(ry.ST.sphere, [0.05]) \
    .setColor([1, 0.5, 0]) \
    .setPosition([target_local[0],  0.6 + target_local[1], target_local[2]])

q_home = C.getJointState()

# 显示初始姿态
C.view(True, "初始: 两个 Panda 都在 q_home 姿态 - 按键开始求解")


# ==================== 2. 左 Panda: 软约束 (μ=0.5) ====================
komo_L = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=0, enableCollisions=False)
# 任务约束: 软的(SOS),scale = sqrt(μ)
komo_L.addObjective([], ry.FS.positionDiff, ['L_gripper', 'L_target'],
                    ry.OT.sos, scale=[np.sqrt(0.5)])
# 正则: ||q - q_home||²
komo_L.addObjective([], ry.FS.qItself, [], ry.OT.sos,
                    scale=[1.0], target=q_home)
ry.NLP_Solver(komo_L.nlp(), verbose=0).solve()
q_L_full = komo_L.getPath()[-1]


# ==================== 3. 右 Panda: 硬约束 (μ→∞) ====================
komo_R = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=0, enableCollisions=False)
# 任务约束: 硬的(EQ),对应公式(3)的伪逆解,在数学上,EQ 等价于 μ=∞ 的极限,这个极限已经被求解器内部处理了(用拉格朗日方法),不需要你给 μ。
komo_R.addObjective([], ry.FS.positionDiff, ['R_gripper', 'R_target'],
                    ry.OT.eq, scale=[1e2])
komo_R.addObjective([], ry.FS.qItself, [], ry.OT.sos,
                    scale=[1.0], target=q_home)
ry.NLP_Solver(komo_R.nlp(), verbose=0).solve()
q_R_full = komo_R.getPath()[-1]


# ==================== 4. 慢动作播放:从 q_home 平滑过渡到最终解 ====================

def animate_to(C, q_start, q_end, steps=60, dt=0.03, label=""):
    """在 q_start 和 q_end 之间线性插值,逐步显示。"""
    for i in range(steps + 1):
        alpha = i / steps  # 0 → 1
        q_mid = (1 - alpha) * q_start + alpha * q_end
        C.setJointState(q_mid)
        C.view(False, f"{label}  进度 {alpha*100:.0f}%")
        time.sleep(dt)


# 把两个解合并到一个完整状态向量
q_combined = q_home.copy()
q_combined[:7] = q_L_full[:7]
q_combined[7:] = q_R_full[7:]

# 慢慢动过去(60 步,每步 30ms,总共约 1.8 秒)
animate_to(C, q_home, q_combined, steps=60, dt=0.03,
           label="左:μ=0.5(软)  |  右:μ→∞(硬)")


# ==================== 5. 计算并打印误差 ====================
err_L = np.linalg.norm(
    C.getFrame('L_target').getPosition() - C.getFrame('L_gripper').getPosition())
err_R = np.linalg.norm(
    C.getFrame('R_target').getPosition() - C.getFrame('R_gripper').getPosition())

dq_L = np.linalg.norm(q_L_full[:7] - q_home[:7])
dq_R = np.linalg.norm(q_R_full[7:] - q_home[7:])

print("="*70)
print("                    IK 双 Panda 对比实验")
print("="*70)
print(f"目标位置(相对于各自 base): {target_local}")
print()
print(f"{'机械臂':>8} | {'类型':>12} | {'末端误差':>14} | {'关节变化(rad)':>14}")
print("-"*70)
print(f"{'左 Panda':>8} | {'μ=0.5 (软)':>12} | {err_L*100:>10.2f} cm   | {dq_L:>14.4f}")
print(f"{'右 Panda':>8} | {'μ→∞ (硬)':>12} | {err_R*1000:>10.3f} mm   | {dq_R:>14.4f}")
print("="*70)

C.view(True, f"左:μ=0.5(差{err_L*100:.0f}cm) | 右:μ→∞(精确) - 按键退出")