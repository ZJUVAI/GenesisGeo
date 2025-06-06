import pandas as pd
import matplotlib.pyplot as plt
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