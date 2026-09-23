# 联合去噪扩散

## 核心思想

ME-Dex 用**单个扩散过程**同时对视频、触觉、动作序列去噪。这样三者在时间上天然
保持一致——「看到的未来」和「摸到的未来」是同一个未来。

## 实现

- `NoiseSchedule`：线性 beta 调度，`beta_start=1e-4`, `beta_end=0.02`
- `JointDenoiser`：给定带噪 token + 条件，预测去噪 token
- 采样：从高斯噪声逐步逆扩散

## 代码

```python
from dexworld.model.diffusion import JointDenoiser, NoiseSchedule

schedule = NoiseSchedule(steps=20)
denoiser = JointDenoiser(dim=64)
denoised = denoiser.sample(tokens, condition=cond, steps=20)
```
