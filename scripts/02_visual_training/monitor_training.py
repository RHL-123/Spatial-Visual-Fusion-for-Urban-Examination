#!/usr/bin/env python3
"""
监控DETR训练进度和GPU内存使用
"""

import os
import time
import subprocess
import psutil
import torch

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free
    }

def get_system_memory_info():
    """获取系统内存信息"""
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / 1024**3,
        "available_gb": mem.available / 1024**3,
        "used_gb": mem.used / 1024**3,
        "percent": mem.percent
    }

def get_process_info(pid):
    """获取进程信息"""
    try:
        process = psutil.Process(pid)
        cpu_percent = process.cpu_percent(interval=1.0)
        memory_info = process.memory_info()
        return {
            "cpu_percent": cpu_percent,
            "rss_gb": memory_info.rss / 1024**3,
            "vms_gb": memory_info.vms / 1024**3,
            "status": process.status()
        }
    except psutil.NoSuchProcess:
        return None

def check_training_progress():
    """检查训练进度"""
    checkpoint_dir = "detr_r50_6_refine_checkpoints"
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        checkpoint_files = [f for f in files if f.endswith('.pth')]
        return {
            "checkpoint_dir": checkpoint_dir,
            "num_checkpoints": len(checkpoint_files),
            "checkpoint_files": checkpoint_files
        }
    return {"checkpoint_dir": checkpoint_dir, "num_checkpoints": 0, "checkpoint_files": []}

def main():
    print("=" * 80)
    print("DETR训练监控系统")
    print("=" * 80)
    
    # 查找训练进程
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'train_detr_r50_6_refine.py' in ' '.join(cmdline):
                training_processes.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not training_processes:
        print("未找到正在运行的DETR训练进程")
        return
    
    print(f"找到 {len(training_processes)} 个训练进程: {training_processes}")
    
    for pid in training_processes:
        print(f"\n{'='*40}")
        print(f"监控进程 PID: {pid}")
        print(f"{'='*40}")
        
        # 获取进程信息
        proc_info = get_process_info(pid)
        if proc_info:
            print(f"CPU使用率: {proc_info['cpu_percent']:.1f}%")
            print(f"内存使用: RSS={proc_info['rss_gb']:.2f}GB, VMS={proc_info['vms_gb']:.2f}GB")
            print(f"进程状态: {proc_info['status']}")
        
        # 获取GPU信息
        gpu_info = get_gpu_memory_info()
        print(f"\nGPU内存: 已分配={gpu_info['allocated_gb']:.2f}GB, "
              f"已保留={gpu_info['reserved_gb']:.2f}GB, "
              f"可用={gpu_info['free_gb']:.2f}GB")
        
        # 获取系统内存信息
        sys_mem = get_system_memory_info()
        print(f"\n系统内存: 总共={sys_mem['total_gb']:.2f}GB, "
              f"可用={sys_mem['available_gb']:.2f}GB, "
              f"使用率={sys_mem['percent']:.1f}%")
    
    # 检查训练进度
    progress = check_training_progress()
    print(f"\n{'='*40}")
    print("训练进度检查")
    print(f"{'='*40}")
    print(f"检查点目录: {progress['checkpoint_dir']}")
    print(f"检查点数量: {progress['num_checkpoints']}")
    if progress['checkpoint_files']:
        print("检查点文件:")
        for f in progress['checkpoint_files']:
            file_path = os.path.join(progress['checkpoint_dir'], f)
            size_mb = os.path.getsize(file_path) / 1024**2
            print(f"  - {f} ({size_mb:.1f} MB)")
    
    # 检查日志文件
    log_files = [f for f in os.listdir('.') if f.endswith('.log')]
    if log_files:
        print(f"\n日志文件: {', '.join(log_files)}")
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"\n{log_file} 最后几行:")
                        for line in lines[-5:]:
                            print(f"  {line.strip()}")
            except Exception as e:
                print(f"读取日志文件 {log_file} 失败: {e}")

if __name__ == "__main__":
    main()
