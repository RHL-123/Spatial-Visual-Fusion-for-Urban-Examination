from ultralytics import RTDETR
import os
from academic_logger import attach_academic_logger

# 确保在脚本所在目录运行
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_RUNS_DIR = os.path.join(BASE_DIR, 'training_runs', 'rtdetr')
# 您只需在此处修改所使用的数据集即可（如 'datasets' 或 'datasets_balanced'）
DATA_YAML = os.path.join(BASE_DIR, '..', 'datasets_balanced', 'data.yaml')

# 提取您使用的数据集文件夹名称（如 "datasets" 或 "datasets_balanced"）
DATASET_NAME = os.path.basename(os.path.dirname(DATA_YAML))
EXP_NAME = f"rtdetr_{DATASET_NAME}"

if __name__ == '__main__':
    # 加载最新的 Vision Transformer 目标检测架构：RT-DETR (Real-Time DEtection TRansformer)
    # 强制指定预训练模型的绝对路径读取，防止触发意外下载
    WEIGHTS_PATH = os.path.join(BASE_DIR, '..', 'rtdetr-l.pt')
    model = RTDETR(WEIGHTS_PATH)
    
    # 挂载学术级日志记录器，捕捉吞吐量、显存和环境等！
    attach_academic_logger(model)

    print(f"===========================================================")
    print(f"🚀 初始化基于 ViT/Transformer 的 RT-DETR 模型...")
    print(f"📂 正在使用的数据集: {DATASET_NAME}")
    print(f"🎯 训练结果及日志将保存在: {os.path.join(TRAINING_RUNS_DIR, EXP_NAME)} 下...")
    print(f"===========================================================")
    
    # 开始训练
    model.train(
        data=DATA_YAML,
        epochs=300,           # Transformer 模型通常收敛较快，这里建议150-300
        imgsz=640,
        batch=8,              # ViT/Transformer 架构占用显存较大，默认先调小到 8，如果显存足够可调回 16
        workers=0, 
        device='1',           # 👈 指定使用的GPU为 1卡
        project=TRAINING_RUNS_DIR,
        name=EXP_NAME,        # 👈 根据您更换的数据集更改保存的输出文件夹名称！
        optimizer='AdamW',    # Transformer 架构强烈建议使用 AdamW 优化器以防止过拟合
        lr0=0.0001            # RT-DETR 的默认学习率远低于普通 YOLO
    )

