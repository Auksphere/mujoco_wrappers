#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def plot_fz_only():
    """
    只绘制Fz力，与期望值对比
    """
    csv_file_path = './log/contact_forces.csv'
    
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return None
    
    # 读取数据
    time = []
    fz = []
    
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                time.append(float(row['time']))
                fz.append(float(row['fz']))
            except (ValueError, KeyError):
                continue
    
    if len(time) == 0:
        print("No valid data found in CSV file")
        return None
    
    time = np.array(time)
    fz = np.array(fz)
    
    # 创建单个图表
    plt.figure(figsize=(12, 6))
    
    # 绘制原始Fz数据
    plt.plot(time, fz, label='Actual Fz', linewidth=2, color='blue')
    
    # 绘制期望值
    desired_fz = 10.0
    plt.axhline(y=desired_fz, color='red', linestyle='--', linewidth=2, 
               label=f'Desired Fz = {desired_fz} N')
    
    # 填充误差区域
    error = fz - desired_fz
    plt.fill_between(time, desired_fz, fz, 
                   where=(fz >= desired_fz), 
                   alpha=0.3, color='red', label='Positive Error')
    plt.fill_between(time, desired_fz, fz, 
                   where=(fz < desired_fz), 
                   alpha=0.3, color='blue', label='Negative Error')
    
    # 设置标签和标题
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Fz (N)', fontsize=12)
    plt.title('Vertical Contact Force (Fz) vs Desired Value', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 计算并显示统计信息
    mean_fz = np.mean(fz)
    std_fz = np.std(fz)
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # 在图上添加统计信息文本
    stats_text = f'Mean: {mean_fz:.2f} N\n'
    stats_text += f'Std: {std_fz:.2f} N\n'
    stats_text += f'RMSE: {rmse:.2f} N\n'
    stats_text += f'Max Error: {max_error:.2f} N'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/fz_tracking.png', dpi=300, bbox_inches='tight')
    print("Plot saved to ./plots/fz_tracking.png")
    
    # 打印统计信息
    print(f"\n=== Fz Tracking Statistics ===")
    print(f"Data points: {len(time)}")
    print(f"Time range: {time[0]:.3f} - {time[-1]:.3f} seconds")
    print(f"Desired Fz: {desired_fz} N")
    print(f"Actual Fz: {mean_fz:.3f} ± {std_fz:.3f} N")
    print(f"Tracking error (RMSE): {rmse:.3f} N")
    print(f"Maximum error: {max_error:.3f} N")
    print(f"Minimum Fz: {np.min(fz):.3f} N")
    print(f"Maximum Fz: {np.max(fz):.3f} N")
    
    return plt.gcf()

if __name__ == "__main__":
    fig = plot_fz_only()
    if fig:
        plt.show()