from ultralytics import YOLO
import os
from academic_logger import attach_academic_logger

# 确保在脚本所在目录运行
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_RUNS_DIR = os.path.join(BASE_DIR, 'training_runs', 'yolov9')
# 依然指向你的平衡数据集
DATA_YAML = os.path.join(BASE_DIR, '..', 'datasets_balanced', 'data.yaml')

DATASET_NAME = os.path.basename(os.path.dirname(DATA_YAML))

# 与原先 imgsz=640 的「yolov9_datasets_balanced」区分：run 名带分辨率后缀
IMGSZ = 1280
# 1280 相对 640：每步算力约×4，且同 batch 下总步数不变时整轮更慢——应优先提高 batch（你之前 bs=4 时仅~10GB/24GB，可试 8）。
# 若 OOM：把 BATCH 改为 6 或 4；若仍嫌慢：把 IMGSZ 改为 960（比 1280 快一截）。
BATCH = 8 if IMGSZ >= 960 else 16
# Linux/Docker 下可开多进程加速 decode；环境变量 YOLO_WORKERS 可覆盖（如 Windows 下设为 0）
WORKERS = int(os.environ.get("YOLO_WORKERS", "4"))


def _parse_cache(s: str | None) -> bool | str:
    v = (s or "").strip().lower()
    if v in ("", "0", "false", "off", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    if v in ("ram", "disk"):
        return v
    return False


# 默认关缓存：cache='disk' 会在 images/train|val 旁写 .npy；中断/多进程时易出现「缺某个 .npy」而报 FileNotFoundError。
# 若要用 disk：先跑完让缓存建全，或删除数据集下所有 *.npy 后重跑。加速可用 YOLO_CACHE=ram（内存够时）。
CACHE = _parse_cache(os.environ.get("YOLO_CACHE", "false"))

EXP_NAME = f"yolov9_{DATASET_NAME}_imgsz{IMGSZ}"

if __name__ == '__main__':
    # YOLOv9 常用的主流规模是 yolov9c.pt (Compact) 和 yolov9e.pt (Extended)
    # 推荐从 yolov9c 开始跑对比实验
    WEIGHTS_PATH = os.path.join(BASE_DIR, 'yolov9c.pt')
    
    # 初始化模型，会下载预训练权重
    model = YOLO(WEIGHTS_PATH)
    
    # 挂载学术日志记录器，精准捕获 PGI 架构的计算开销
    attach_academic_logger(model)

    print(f"===========================================================")
    print(f"🚀 初始化 YOLOv9 (PGI + GELAN) 目标检测模型...")
    print(f"📂 正在使用的数据集: {DATASET_NAME}")
    print(f"📐 imgsz={IMGSZ}, batch={BATCH}, workers={WORKERS}, cache={CACHE}")
    print(f"🎯 训练结果及日志将保存在: {os.path.join(TRAINING_RUNS_DIR, EXP_NAME)} 下...")
    print(f"===========================================================")
    
    model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        cache=CACHE,
        device='1',
        project=TRAINING_RUNS_DIR, 
        name=EXP_NAME,
        optimizer='auto',     # 官方默认优化器策略通常最优
        close_mosaic=10       # YOLOv9 强烈建议在最后 10 个 epoch 关闭马赛克增强，有利于提升精度
    )

