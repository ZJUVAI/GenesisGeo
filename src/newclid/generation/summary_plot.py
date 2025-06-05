import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
def summary_plot(summaries: list[dict[str, float]], prefix: str = ''):
    df = pd.DataFrame(summaries) 
    logging.info(df.describe())

    # 创建第一个图：time的分布
    plt.figure(figsize=(10, 6))
    plt.hist(df['runtime'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.title('Time Distribution')
    plt.grid(True, alpha=0.3)
    # 添加统计信息
    mean_time = df['runtime'].mean()
    median_time = df['runtime'].median()
    plt.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}s')
    plt.axvline(median_time, color='orange', linestyle='--', label=f'Median: {median_time:.2f}s')
    plt.legend()
    # 保存第一个图
    plt.tight_layout()
    plt.savefig(f'{prefix}_summary_time.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    # 创建第一个图：N-samples的分布
    plt.figure(figsize=(10, 6))
    plt.hist(df['n_samples'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('#Samples')
    plt.ylabel('Frequency')
    plt.title('#Samples Distribution')
    plt.grid(True, alpha=0.3)
    # 添加统计信息
    mean = df['n_samples'].mean()
    median = df['n_samples'].median()
    plt.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median_time, color='orange', linestyle='--', label=f'Median: {median:.2f}')
    plt.legend()
    # 保存第一个图
    plt.tight_layout()
    plt.savefig(f'{prefix}_summary_nsamples.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    # # 创建第二个图：x-y轴调转，samples_per_thread和ddar_time的关系
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df['ddar_time'], df['samples_per_thread'], 
    #         alpha=0.7, color='red', s=60)
    # plt.xlabel('DDAR Time (s)')
    # plt.ylabel('Samples per Thread')
    # plt.title('Samples per Thread vs DDAR Time')
    # plt.grid(True, alpha=0.3)
    # # 计算统计信息
    # num_ddar_samples = len(df)
    # total_samples = df['samples_per_thread'].sum()
    # max_samples = df['samples_per_thread'].max()
    # min_samples = df['samples_per_thread'].min()
    # mean_samples = df['samples_per_thread'].mean()
    # # 添加英文标注
    # stats_text = f'DDAR Samples: {num_ddar_samples}\n'
    # stats_text += f'Total Samples: {total_samples}\n'
    # stats_text += f'Max Samples per DDAR: {max_samples}\n'
    # stats_text += f'Min Samples per DDAR: {min_samples}\n'
    # stats_text += f'Mean Samples per DDAR: {mean_samples:.2f}'
    # # 在图的右上角添加文本框
    # plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
    #         verticalalignment='top', horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    #         fontsize=10)
    # # 保存第二个图
    # plt.tight_layout()
    # plt.savefig('./dataset/time_samples_relation.jpg', dpi=300, bbox_inches='tight')
    # plt.close()

    







# # 打印一些基本统计信息
# print("Data Summary:")
# print(f"Total samples: {len(df)}")
# print(f"DDAR Time - Mean: {df['ddar_time'].mean():.2f}s")
# print(f"DDAR Time - Median: {df['ddar_time'].median():.2f}s")
# print(f"DDAR Time - Std Dev: {df['ddar_time'].std():.2f}s")
# print(f"DDAR Time - Range: {df['ddar_time'].min():.2f}s - {df['ddar_time'].max():.2f}s")
# print(f"Samples per Thread - Mean: {df['samples_per_thread'].mean():.2f}")
# print(f"Samples per Thread - Median: {df['samples_per_thread'].median():.2f}")
# print(f"Correlation coefficient: {df['ddar_time'].corr(df['samples_per_thread']):.4f}")

# print("Images saved:")
# print("- DDAR Time Distribution: ./dataset/ddar_time_distribution.jpg")
# print("- Time vs Samples Relation: ./dataset/time_samples_relation.jpg")