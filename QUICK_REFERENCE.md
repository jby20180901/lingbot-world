# 快速参考 - LingBot World 滑动窗口

## ⚡ 1分钟快速开始

### 启用滑动窗口生成200帧视频
```bash
python generate.py \
    --task i2v-A14B \
    --ckpt_dir path/to/checkpoints \
    --image input.jpg \
    --prompt "A cinematic video..." \
    --frame_num 200 \
    --use_sliding_window
```

## 📊 显存对比

| 场景 | 显存 |
|------|------|
| 49帧 (无窗口) | ~5GB |
| 97帧 (无窗口) | ~10GB ❌ |
| **97帧 (窗口)** | **~5GB** ✓ |
| 193帧 (无窗口) | ~20GB ❌ |
| **193帧 (窗口)** | **~5GB** ✓ |

## 🎛️ 关键参数

```
--use_sliding_window          # 启用功能
--window_size 49              # 窗口大小 (49, 57, 81, ...)
--overlap_size 24             # 重叠大小
--blend_mode lerp             # 混合方式 (lerp 或 crossfade)
```

## 🎯 配置建议

| 显存 | 配置 |
|------|------|
| < 6GB | `--window_size 49 --overlap_size 24 --blend_mode lerp` |
| 6-10GB | `--window_size 57 --overlap_size 28 --blend_mode crossfade` |
| > 10GB | `--window_size 81 --overlap_size 40 --blend_mode crossfade` |

## 🚀 常用命令

### 低显存设备
```bash
python generate.py \
    --task i2v-A14B \
    --ckpt_dir checkpoints \
    --image input.jpg \
    --prompt "..." \
    --frame_num 300 \
    --use_sliding_window \
    --offload_model
```

### 最高质量
```bash
python generate.py \
    --task i2v-A14B \
    --ckpt_dir checkpoints \
    --image input.jpg \
    --prompt "..." \
    --frame_num 300 \
    --use_sliding_window \
    --window_size 81 \
    --overlap_size 40 \
    --blend_mode crossfade
```

## 💡 Python API

```python
from wan.sliding_window_i2v import SlidingWindowI2V

sw = SlidingWindowI2V(wan_i2v_model, window_size=49)
video = sw.generate_sliding_window(
    img=image,
    input_prompt="...",
    frame_num=200,
    seed=42
)
```

## ⚠️ 常见问题

| 问题 | 解决方案 |
|------|----------|
| 显存不足 | 减小 window_size |
| 视频闪烁 | 增加 overlap_size 或改为 crossfade |
| 相机抖动 | 增加重叠，改进轨迹质量 |
| 生成缓慢 | 减小 window_size 或 sample_steps |

## 📚 详细文档

- `SLIDING_WINDOW_GUIDE.md` - 完整使用指南
- `SLIDING_WINDOW_IMPLEMENTATION.md` - 实现细节
- `test_sliding_window.py` - 测试脚本

## 📈 性能数据

生成时间 ≈ `ceil(总帧数 / 窗口大小) × 基准时间`

例：
- 49帧: 1 × 基准 ≈ 30秒
- 97帧: 2 × 基准 ≈ 60秒
- 145帧: 3 × 基准 ≈ 90秒

## ✓ 文件清单

- [x] `wan/sliding_window_i2v.py` - 核心实现
- [x] `wan/__init__.py` - 导出类
- [x] `generate.py` - 集成参数
- [x] `SLIDING_WINDOW_GUIDE.md` - 详细文档
- [x] `SLIDING_WINDOW_IMPLEMENTATION.md` - 实现文档
- [x] `test_sliding_window.py` - 测试
- [x] `demo_sliding_window.py` - 演示
- [x] `QUICK_REFERENCE.md` - 本文件

## 🔗 相关资源

- HY-WorldPlay 无限视频生成: `generate_infinite_video.py`
- Flow Matching 调度器: `wan/utils/fm_solvers.py`
- VAE 编码器: `wan/modules/vae2_1.py`
