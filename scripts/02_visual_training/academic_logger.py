import os
import time
import torch
import json
import subprocess
import sys

try:
    import psutil
except ImportError:
    psutil = None

try:
    from thop import profile
except ImportError:
    profile = None

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except:
        return "N/A (Not a git repository)"

def get_gpu_utilization(device_id):
    try:
        dev_id = str(device_id).split(',')[0].replace('cuda:', '').strip()
        if not dev_id: dev_id = '0'
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', dev_id], 
            stderr=subprocess.DEVNULL
        )
        return float(result.decode('utf-8').strip())
    except:
        return 0.0

class AcademicTracker:
    def __init__(self):
        self.epoch_start_time = 0
        self.device_id = '0'

    def on_train_start(self, trainer):
        self.train_start_time = time.time()
        save_dir = trainer.save_dir
        
        # 解析 GPU 设备
        device = trainer.device
        self.device_id = str(device.index) if hasattr(device, 'index') and device.index is not None else '0'
        
        gpu_name = "CPU"
        if torch.cuda.is_available() and device.type != 'cpu':
            try:
                gpu_name = torch.cuda.get_device_name(device)
            except:
                pass
                
        # 计算 FLOPs 和 Params
        flops_g, params_m = "N/A", "N/A"
        if hasattr(trainer, 'model'):
            # 统计参数量 (M=百万)
            total_params = sum(p.numel() for p in trainer.model.parameters())
            params_m = round(total_params / 1e6, 2)
            
            # 使用 thop 统计 FLOPs (G=十亿)
            if profile is not None:
                try:
                    import copy
                    imgsz = getattr(trainer.args, 'imgsz', 640)
                    if isinstance(imgsz, int): imgsz = (imgsz, imgsz)
                    dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1]).to(device)
                    # 将模型复制一份给 thop 剖析，防止 thop 在原模型上打的 hooks 污染原模型导致训练报错
                    flops_model = copy.deepcopy(trainer.model)
                    macs, _ = profile(flops_model, inputs=(dummy_input, ), verbose=False)
                    flops_g = round(macs * 2 / 1e9, 2)  # MACs to FLOPs
                    del flops_model # 及时释放显存
                except Exception as e:
                    flops_g = f"Error: {str(e)}"
            else:
                flops_g = "N/A (Please `pip install thop`)"

        # 1. 记录学术论文需要的元数据 (Metadata)
        meta = {
            "Python_Version": sys.version.split(' ')[0],
            "PyTorch_Version": torch.__version__,
            "CUDA_Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "GPU_Name": gpu_name,
            "GPU_Device_Index": self.device_id,
            "Code_Commit": get_git_commit(),
            "Parameters_M": params_m,
            "FLOPs_G": flops_g,
            "Random_Seed": getattr(trainer.args, 'seed', 'N/A'),
            "Config": vars(trainer.args) if hasattr(trainer.args, '__dict__') else str(trainer.args)
        }
        
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'academic_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=4, default=str)
            
        # 2. 初始化用于记录计算效率和复杂度的 CSV 文件
        self.csv_path = os.path.join(save_dir, 'academic_metrics.csv')
        with open(self.csv_path, 'w') as f:
            f.write("Epoch,Time_sec,Throughput_samples_per_sec,GPU_Max_Memory_MB,GPU_Util_percent,RAM_Usage_MB\n")

    def on_train_epoch_start(self, trainer):
        self.epoch_start_time = time.time()
        # 每个 Epoch 开始前，重置显存峰值统计
        if torch.cuda.is_available() and trainer.device.type != 'cpu':
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def on_train_epoch_end(self, trainer):
        epoch_time = time.time() - self.epoch_start_time
        
        # 计算吞吐量 (Throughput: Samples/sec)
        total_samples = 0
        try:
            total_samples = len(trainer.train_loader.dataset)
        except:
            pass
        throughput = total_samples / epoch_time if epoch_time > 0 else 0
        
        # 记录显存和 GPU 利用率
        gpu_mem = 0
        if torch.cuda.is_available() and trainer.device.type != 'cpu':
            gpu_mem = torch.cuda.max_memory_allocated(trainer.device) / (1024 * 1024)
            
        gpu_util = get_gpu_utilization(self.device_id)
        
        # 记录内存 (RAM)
        ram_usage = 0
        if psutil:
            ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        epoch_num = trainer.epoch if hasattr(trainer, 'epoch') else 0
        with open(self.csv_path, 'a') as f:
            f.write(f"{epoch_num},{epoch_time:.2f},{throughput:.2f},{gpu_mem:.2f},{gpu_util:.2f},{ram_usage:.2f}\n")

def attach_academic_logger(model):
    """
    Attach academic logger to the model.
    Works with both standard Ultralytics models and custom models like LaSt-ViT.
    """
    tracker = AcademicTracker()
    
    # Check if the model has callback functionality (standard Ultralytics models)
    if hasattr(model, 'add_callback'):
        model.add_callback("on_train_start", tracker.on_train_start)
        model.add_callback("on_train_epoch_start", tracker.on_train_epoch_start)
        model.add_callback("on_train_epoch_end", tracker.on_train_epoch_end)
    else:
        # For custom models like LaSt-ViT, we'll need to integrate differently
        # This is a fallback that logs a warning
        print("⚠️  Model does not support callbacks. Academic logger may not work properly with this custom model.")
        print("💡 Consider integrating academic logging directly into your custom model.")

