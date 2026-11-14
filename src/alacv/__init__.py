"""ALAC-V Phase 1: visual-first static image encoding pipeline.

Modules
-------
config      : 配置加载、校验、快照与哈希
logging_utils: 结构化 JSONL + 控制台日志
ingest      : 只读、内容寻址的原始图像归档
imaging     : 解码、RGB 模式归一（不做 ICC 变换）、多尺度金字塔
foveation   : ROI 选择与 foveated 合成
features    : DINOv2 ViT-S/14 冻结编码器（CLS + Patch, FP16）
store       : SQLite 元信息 + SafeTensors 特征
retrieval   : 描述子检索与高清回溯
metrics     : 计时 / 显存 / CPU / RAM / PSNR / SSIM
pipeline    : 变体 A/B/C 的编排
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
