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
EXP_NAME = f"rtdetr_{DATASET_NAME}_tech3"

if __name__ == '__main__':
    # 加载最新的 Vision Transformer 目标检测架构：RT-DETR (Real-Time DEtection TRansformer)
    # 强制指定预训练模型的绝对路径读取，防止触发意外下载
    WEIGHTS_PATH = os.path.join(BASE_DIR, '..', 'rtdetr-l.pt')
    model = RTDETR(WEIGHTS_PATH)
    
    # 挂载学术级日志记录器，捕捉吞吐量、显存和环境等！
    attach_academic_logger(model)

    print(f"===========================================================")
    print(f"🚀 初始化带【抗极度不均衡四联装增强】的 ViT/RT-DETR 模型...")
    print(f"📂 正在使用的数据集: {DATASET_NAME}")
    print(f"🛠️ 已开启: 复制粘贴 (copy_paste), 图像混合 (mixup), 马赛克 (mosaic), 随机遮挡 (erasing)!")
    print(f"🎯 训练结果及日志将保存在: {os.path.join(TRAINING_RUNS_DIR, EXP_NAME)} 下...")
    print(f"===========================================================")
    
    # 开始训练
    model.train(
        data=DATA_YAML,
        epochs=300,           # 考虑到欠采样后数据集变小，我们可以放心地跑到 300 轮
        imgsz=640,
        batch=8,              # ViT 框架大，默认先 8 ，以防爆显存
        workers=0, 
        device='1',           # 👈 指定使用的主力GPU为 1卡
        project=TRAINING_RUNS_DIR, 
        name=EXP_NAME,        # 避免和之前的普通版覆盖
        optimizer='AdamW',
        lr0=0.0001,
        
        # 🌟🌟🌟 解决严重类别不平衡的【三大黑科技】参数 🌟🌟🌟
        # (注：最新 Ultralytics 取消了 image_weights 和 fl_gamma 直接参数暴露，改为更强的数据增强组合)
        copy_paste=0.3,       # 1. 内存复制粘贴大法：以 30% 的概率，把稀有类的框抠出来，生硬地贴到毫无关系的图上（极大丰富多样性）
        mixup=0.1,            # 2. 图像混合：按比例融合两张图片，降低模型对绝对背景的依赖
        mosaic=1.0,           # 3. 马赛克增强：四图拼接，让小物体的上下文背景随机变化，提高小目标稀有物体鲁棒性
        erasing=0.4           #    随机遮挡：迫使模型通过局部特征识别（管道）
    )

