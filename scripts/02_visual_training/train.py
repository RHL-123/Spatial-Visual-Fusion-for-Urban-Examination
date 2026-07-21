from ultralytics import YOLO
import os
from academic_logger import attach_academic_logger

# 确保在脚本所在目录运行
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_RUNS_DIR = os.path.join(BASE_DIR, 'training_runs', 'yolov8')
# 您只需在此处修改所使用的数据集即可（如 'datasets' 或 'datasets_balanced'）
DATA_YAML = os.path.join(BASE_DIR, '..', 'datasets_balanced', 'data.yaml')

# 提取您使用的数据集文件夹名称（如 "datasets" 或 "datasets_balanced"）
DATASET_NAME = os.path.basename(os.path.dirname(DATA_YAML))
EXP_NAME = f"yolov8_{DATASET_NAME}"

if __name__ == '__main__':
    # 加载模型 (yolov8n.pt, yolov8s.pt, yolov8m.pt 等)
    # 如果没有预训练权重，会下载 (这里改为绝对路径读取)
    WEIGHTS_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')
    model = YOLO(WEIGHTS_PATH) 
    
    # 挂载学术级日志记录器，捕捉吞吐量、显存和环境等！
    attach_academic_logger(model)

    # 开始训练
    # data: 数据集配置文件路径
    # epochs: 训练轮数
    # imgsz: 图片大小
    # batch: batch size (根据显存调整, -1为)
    print(f"===========================================================")
    print(f"🚀 初始化 YOLOv8 目标检测模型...")
    print(f"📂 正在使用的数据集: {DATASET_NAME}")
    print(f"🎯 训练结果及日志将保存在: {os.path.join(TRAINING_RUNS_DIR, EXP_NAME)} 下...")
    print(f"===========================================================")
    
    # 注意：Windows下使用多进程可能会有问题，workers设置为0或1可以避免
    model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=640,
        batch=16,
        workers=0, 
        device='0',     # 👈 指定使用的GPU为 0卡
        project=TRAINING_RUNS_DIR,
        name=EXP_NAME  # 👈 自适应不同数据集生成的文件夹
    )

