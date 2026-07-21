from ultralytics import YOLO
import os
from academic_logger import attach_academic_logger

# 确保在脚本所在目录运行
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_RUNS_DIR = os.path.join(BASE_DIR, 'training_runs', 'yolo11')
# 指向你的平衡数据集
DATA_YAML = os.path.join(BASE_DIR, '..', 'datasets_balanced', 'data.yaml')

DATASET_NAME = os.path.basename(os.path.dirname(DATA_YAML))
EXP_NAME = f"yolo11_{DATASET_NAME}"

if __name__ == '__main__':
    # 注意命名变化：官方去掉了 'v'，直接叫 yolo11
    # 强制指定本地绝对路径，防止触发 SSL 报错
    WEIGHTS_PATH = os.path.join(BASE_DIR, 'yolo11n.pt')
    
    model = YOLO(WEIGHTS_PATH)
    
    # 挂载学术日志记录器，重点观察它极低的 FLOPs_G 和 Parameters_M
    attach_academic_logger(model)

    print(f"===========================================================")
    print(f"🚀 初始化 YOLO11 (含 C3k2 & C2PSA 注意力) 目标检测模型...")
    print(f"📂 正在使用的数据集: {DATASET_NAME}")
    print(f"🎯 训练结果及日志将保存在: {os.path.join(TRAINING_RUNS_DIR, EXP_NAME)} 下...")
    print(f"===========================================================")
    
    model.train(
        data=DATA_YAML,
        epochs=300,           # 保持 300 轮，与 v8/v9/v10 绝对公平对比
        imgsz=640,
        batch=16,             # 参数量减少了，RTX 3090 跑 batch=16 毫无压力，甚至可以上 32
        workers=0,            # Docker/Windows 环境防报错
        device='0',
        project=TRAINING_RUNS_DIR, 
        name=EXP_NAME,
        close_mosaic=10       # 保持一致的数据增强关闭策略
    )

