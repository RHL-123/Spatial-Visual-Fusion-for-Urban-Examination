from ultralytics import YOLO
import os
from academic_logger import attach_academic_logger

# 确保在脚本所在目录运行
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_RUNS_DIR = os.path.join(BASE_DIR, 'training_runs', 'yolov10')
DATA_YAML = os.path.join(BASE_DIR, '..', 'datasets_balanced', 'data.yaml')

DATASET_NAME = os.path.basename(os.path.dirname(DATA_YAML))
EXP_NAME = f"yolov10_{DATASET_NAME}"

if __name__ == '__main__':
    # YOLOv10 提供 n, s, m, b, l, x 等多个版本
    # 这里使用 yolov10s.pt 对齐常规的对比实验标准
    WEIGHTS_PATH = os.path.join(BASE_DIR, 'yolov10s.pt')
    model = YOLO(WEIGHTS_PATH) 
    
    # 挂载学术日志记录器
    attach_academic_logger(model)

    print(f"===========================================================")
    print(f"🚀 初始化 YOLOv10 (无 NMS 端到端) 目标检测模型...")
    print(f"📂 正在使用的数据集: {DATASET_NAME}")
    print(f"🎯 训练结果及日志将保存在: {os.path.join(TRAINING_RUNS_DIR, EXP_NAME)} 下...")
    print(f"===========================================================")
    
    model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=640,
        batch=16,
        workers=0, 
        device='1',
        project=TRAINING_RUNS_DIR,
        name=EXP_NAME,
        close_mosaic=10      # 同样建议在训练尾声关闭马赛克数据增强
    )

