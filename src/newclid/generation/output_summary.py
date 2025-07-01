import pandas as pd
import matplotlib.pyplot as plt
import logging
from collections import Counter

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

def plot_goal_distribution(summaries: list[dict[str, float]], prefix: str, ylog: bool = True):
    df = pd.DataFrame(summaries) 
    logging.info(df.describe())
    
    # Extract and flatten the list of goals from all summaries
    if 'goals' not in df.columns or df['goals'].empty:
        logging.warning("The 'goals' column is missing or empty in summaries. Cannot plot goal distribution.")
        goals = []
    else:
        goals = df['goals'].explode().tolist()
        # Ensure all items are strings, handle potential None values if necessary
        goals = [str(g) for g in goals if g is not None]


    if not goals:
        print("No goals found to plot.")
        return

    counter = Counter(goals)
    total = len(goals)

    percentages = {k: (v / total) * 100 for k, v in counter.items()}
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    percs = [percentages[label] for label in labels]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, counts, color='skyblue', log=ylog)

    for bar, perc in zip(bars, percs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{perc:.1f}%', ha='center', va='bottom')

    plt.xlabel('Goal Type')
    plt.ylabel('Count (log scale)' if ylog else 'Count')
    plt.title('Distribution of Geometric Problem Goals')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_image_path = f'{prefix}_goal_distribution.jpg'
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    logging.info(f"Goal distribution plot saved to {output_image_path}")

def get_first_predicate(fl_statement: str):
    if '?' in fl_statement:
        predicates_part, _ = fl_statement.split('?', 1)
    else:
        predicates_part = fl_statement
    
    first_predicate = ''
    try:
        first_statement = predicates_part.split(';')[0].strip()
        predicate_part = first_statement
        if '=' in first_statement:
            predicate_part = first_statement.split('=', 1)[1].strip()
        first_predicate = predicate_part.split(' ')[0]
        return first_predicate
    except IndexError:
        logging.warning(f"Could not parse first predicate for: {fl_statement}")
        return None


def plot_first_predicate_distribution(summaries: list[dict[str, float]], prefix: str, ylog: bool = True):
    df = pd.DataFrame(summaries) 
    
    if 'first_predicate' not in df.columns or df['first_predicate'].empty:
        logging.warning("The 'first_predicate' column is missing or empty in summaries. Cannot plot first predicate distribution.")
        predicates = []
    else:
        predicates = df['first_predicate'].dropna().tolist()

    if not predicates:
        print("No first predicates found to plot.")
        return

    counter = Counter(predicates)
    total = len(predicates)

    percentages = {k: (v / total) * 100 for k, v in counter.items()}
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    percs = [percentages[label] for label in labels]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, counts, color='skyblue', log=ylog)

    for bar, perc in zip(bars, percs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{perc:.1f}%', ha='center', va='bottom')

    plt.xlabel('Predicate Type')
    plt.ylabel('Count (log scale)' if ylog else 'Count')
    plt.title('Distribution of First Predicates')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_image_path = f'{prefix}_first_predicate_distribution.jpg'
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    logging.info(f"First predicate distribution plot saved to {output_image_path}")