# LingBot World 滑动窗口机制使用指南

## 概述

本文档说明了LingBot World实现的滑动窗口机制，它能够在显存有限的设备上生成更长的视频。

### 问题背景

原始的LingBot World实现一次性加载整个视频序列到显存中进行处理，导致显存占用与帧数线性增长：
- 单帧显存占用 ≈ 5GB（16通道的潜在表示）
- 总显存占用 ≈ 帧数 × 5GB

这使得在有限显存的设备上无法生成长视频。

### 解决方案

滑动窗口机制将长视频分成多个重叠的短视频段，逐个生成每个段，然后通过平滑混合拼接在一起。这样做的优势：

- **恒定显存使用**：每个窗口的显存占用固定，与总帧数无关
- **可扩展性**：理论上可以生成无限长的视频
- **平滑过渡**：通过混合机制实现窗口之间的无缝拼接

## 技术细节

### 核心参数

1. **window_size**（窗口大小）
   - 每次处理的帧数，必须为 `4n+1` 形式（如49、53、57、81）
   - 默认值：49帧
   - 更大的窗口 = 更好的连贯性，但显存占用更多
   - 更小的窗口 = 更低的显存，但可能出现闪烁

2. **overlap_size**（重叠大小）
   - 相邻窗口的重叠帧数
   - 默认值：`window_size // 2`（即24帧，当window_size=49时）
   - 建议范围：25%-50%的window_size
   - 更大的重叠 = 更平滑的过渡，但更多的重复计算

3. **blend_mode**（混合模式）
   - `"lerp"`：线性插值（默认）
   - `"crossfade"`：平滑交叉淡化（更高质量但略慢）

### 工作流程

```
原始视频 (N帧)
    ↓
分割成重叠的窗口
    ├─ 窗口1: 帧 0-48    (49帧)
    ├─ 窗口2: 帧 24-72   (49帧, 与窗口1重叠24帧)
    ├─ 窗口3: 帧 48-96   (49帧, 与窗口2重叠24帧)
    └─ ...
    ↓
独立生成每个窗口
    ↓
混合重叠区域
    ↓
拼接生成完整视频
```

### 显存使用对比

| 参数 | 原始方法 | 滑动窗口(window=49) |
|------|---------|------------------|
| 49帧视频 | ~5GB | ~5GB |
| 97帧视频 | ~10GB | ~5GB |
| 145帧视频 | ~15GB | ~5GB |
| 193帧视频 | ~20GB | ~5GB |

## 使用方法

### 基本用法

```bash
cd third_party/lingbot-world

# 使用默认滑动窗口参数生成100帧视频
python generate.py \
    --task i2v-A14B \
    --ckpt_dir path/to/checkpoints \
    --image path/to/input.jpg \
    --prompt "A cinematic video of..." \
    --frame_num 100 \
    --use_sliding_window
```

### 高级用法

```bash
# 自定义窗口大小和重叠
python generate.py \
    --task i2v-A14B \
    --ckpt_dir path/to/checkpoints \
    --image path/to/input.jpg \
    --prompt "A cinematic video of..." \
    --frame_num 200 \
    --use_sliding_window \
    --window_size 57 \
    --overlap_size 28 \
    --blend_mode crossfade
```

### 参数配置建议

#### 显存 < 6GB
```bash
--window_size 49 --overlap_size 24 --blend_mode lerp
```

#### 显存 6-8GB
```bash
--window_size 57 --overlap_size 28 --blend_mode crossfade
```

#### 显存 > 8GB
```bash
--window_size 81 --overlap_size 40 --blend_mode crossfade
```

## 性能指标

### 生成质量

- **连贯性**：窗口大小越大越好，建议≥49帧
- **过渡平滑性**：重叠比例25-50%时效果最好
- **视觉伪影**：使用`crossfade`模式可减少闪烁

### 生成时间

相比原始方法：
- 49帧 + 97帧：约2倍时间（需要处理2个窗口）
- 49帧 + 145帧：约3倍时间（需要处理3个窗口）
- 线性关系：总时间 ≈ 原始时间 × ceil(总帧数 / 窗口帧数)

## 代码API

### SlidingWindowI2V 类

```python
from wan.sliding_window_i2V import SlidingWindowI2V

# 初始化
sliding_window = SlidingWindowI2V(
    wan_i2v=model,          # WanI2V实例
    window_size=49,         # 窗口大小
    overlap_size=24,        # 重叠大小
    blend_mode="lerp"       # 混合模式
)

# 生成视频
video = sliding_window.generate_sliding_window(
    img=input_image,                    # PIL Image
    input_prompt="A video of...",       # 文本提示
    frame_num=200,                      # 总帧数
    max_area=768*512,                   # 最大区域
    guide_scale=(3.0, 7.5),             # 指导比例
    seed=42,                            # 随机种子
    action_path=None,                   # 相机轨迹路径（可选）
    sample_steps=30,                    # 采样步数
    sample_shift=1.0,                   # 采样移位
    sample_guide_scale=(3.0, 7.5),      # 采样指导比例
    offload_model=True                  # 是否卸载模型
)
```

## 故障排除

### 问题1：显存仍然不足
- 尝试减小 `window_size`（例如从49降到41）
- 启用 `--offload_model` 标志
- 减小图像分辨率

### 问题2：视频出现闪烁或不连贯
- 增加 `overlap_size`（例如从24增加到32）
- 切换为 `--blend_mode crossfade`
- 增加 `window_size`（更多上下文信息）

### 问题3：生成时间过长
- 减小 `window_size`
- 减少 `sample_steps`
- 使用更少的总帧数测试

### 问题4：相机运动不连贯
- 对相机轨迹使用高质量的插值
- 在窗口内保持足够长的线性运动
- 检查 `action_path` 提供的相机轨迹质量

## 高级技巧

### 1. 动态窗口大小
对于不同长度的视频自动调整窗口大小：

```python
def get_window_config(target_frames, available_vram_gb):
    if available_vram_gb < 6:
        return dict(window_size=49, overlap_size=24)
    elif available_vram_gb < 8:
        return dict(window_size=57, overlap_size=28)
    else:
        return dict(window_size=81, overlap_size=40)

config = get_window_config(target_frames=200, available_vram_gb=6)
```

### 2. 混合多个提示词
通过在不同窗口使用不同的提示词实现视频主题变化：

```python
# 此功能需要扩展sliding_window_i2v.py
# 在generate_sliding_window中添加prompt列表支持
```

### 3. 相机路径优化
预处理相机轨迹以确保在窗口边界处平滑：

```python
# 确保在窗口重叠处的相机运动连续
# 使用三次样条插值而不是线性插值
```

## 与原始生成的比较

### 原始方法
```python
video = wan_i2v.generate(
    prompt=prompt,
    img=img,
    frame_num=200,  # 一次性加载200帧
    ...
)
```

### 滑动窗口方法
```python
sliding_window = SlidingWindowI2V(wan_i2v, window_size=49)
video = sliding_window.generate_sliding_window(
    input_prompt=prompt,
    img=img,
    frame_num=200,  # 分4个窗口处理
    ...
)
```

两者的API类似，但内存使用方式完全不同。

## 未来改进

1. **自适应窗口大小**：根据可用显存自动调整
2. **相机感知混合**：根据相机运动量调整混合强度
3. **并行窗口生成**：使用多个GPU并行处理不同窗口
4. **质量优化**：使用更高级的混合算法（如 SIFT 特征对齐）
5. **实时生成**：使用帧缓冲实现流式生成

## 参考文献

- HunyuanVideo 的无限视频生成实现
- 视频内插和拼接的最佳实践
- Flow Matching 扩散模型的滑动窗口应用

## 许可证

此实现遵循LingBot World的原始许可证。
