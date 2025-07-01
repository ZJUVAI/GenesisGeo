import json
import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_first_predicate(fl_statement: str):
    if '?' in fl_statement:
        predicates_part, _ = fl_statement.split('?', 1)
    else:
        predicates_part = fl_statement
    
    first_predicate = ''
    try:
        first_statement = predicates_part.split(' ')[0].strip()
        predicate_part = first_statement
        if '=' in first_statement:
            predicate_part = first_statement.split('=', 1)[1].strip()
        first_predicate = predicate_part.split(' ')[0]
        return first_predicate
    except IndexError:
        logging.warning(f"Could not parse first predicate for: {fl_statement}")
        return None

class Summary:
    def __init__(self, prefix: str = ''):
        self.prefix = prefix
        self.summaries = []
        self.df = None
        self.total_elapsed_time = 0
        self.total_samples_generated = 0

    def load_summaries(self, summaries: list[dict[str, any]]):
        self.summaries.extend(summaries)
        self.df = pd.DataFrame(self.summaries)
        logging.info(self.df.describe())

    def add(self, summary: dict[str, any]):
        self.summaries.append(summary)
        self.df = pd.DataFrame(self.summaries)
        # logging.info(self.df.describe())

    def _plot_time_distribution(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['runtime'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency')
        plt.title('Time Distribution')
        plt.grid(True, alpha=0.3)
        mean_time = self.df['runtime'].mean()
        median_time = self.df['runtime'].median()
        plt.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}s')
        plt.axvline(median_time, color='orange', linestyle='--', label=f'Median: {median_time:.2f}s')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.prefix}_summary_time.jpg', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_nsamples_distribution(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['n_samples'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
        plt.xlabel('#Samples')
        plt.ylabel('Frequency')
        plt.title('#Samples Distribution')
        plt.grid(True, alpha=0.3)
        mean = self.df['n_samples'].mean()
        median = self.df['n_samples'].median()
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(median, color='orange', linestyle='--', label=f'Median: {median:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.prefix}_summary_nsamples.jpg', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_proof_length_distribution(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['n_proof_steps'], bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
        plt.xlabel('Proof Length (steps)')
        plt.ylabel('Frequency')
        plt.title('Proof Length Distribution')
        plt.grid(True, alpha=0.3)
        mean_len = self.df['n_proof_steps'].mean()
        median_len = self.df['n_proof_steps'].median()
        plt.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.2f}')
        plt.axvline(median_len, color='orange', linestyle='--', label=f'Median: {median_len:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.prefix}_summary_proof_length.jpg', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_clauses_num_distribution(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['n_clauses'], bins=10, edgecolor='black', alpha=0.7, color='lightcoral')
        plt.xlabel('# Clauses')
        plt.ylabel('Frequency')
        plt.title('# Clauses Distribution')
        plt.grid(True, alpha=0.3)
        mean_clauses = self.df['n_clauses'].mean()
        median_clauses = self.df['n_clauses'].median()
        plt.axvline(mean_clauses, color='red', linestyle='--', label=f'Mean: {mean_clauses:.2f}')
        plt.axvline(median_clauses, color='orange', linestyle='--', label=f'Median: {median_clauses:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.prefix}_summary_clauses_num.jpg', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_goal_distribution(self, ylog: bool = True):
        if 'goals' not in self.df.columns or self.df['goals'].empty:
            logging.warning("The 'goals' column is missing or empty in summaries. Cannot plot goal distribution.")
            return

        goals = self.df['goals'].explode().dropna().tolist()
        goals = [str(g) for g in goals]

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
        bars = plt.bar(labels, counts, color='plum', log=ylog)
        for bar, perc in zip(bars, percs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height, f'{perc:.1f}%', ha='center', va='bottom')

        plt.xlabel('Goal Type')
        plt.ylabel('Count (log scale)' if ylog else 'Count')
        plt.title('Distribution of Geometric Problem Goals')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_image_path = f'{self.prefix}_goal_distribution.jpg'
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        logging.info(f"Goal distribution plot saved to {output_image_path}")

    def _plot_first_predicate_distribution(self, ylog: bool = True):
        if 'first_predicate' not in self.df.columns or self.df['first_predicate'].empty:
            logging.warning("The 'first_predicate' column is missing or empty. Cannot plot first predicate distribution.")
            return

        predicates = self.df['first_predicate'].dropna().tolist()

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
        bars = plt.bar(labels, counts, color='aquamarine', log=ylog)
        for bar, perc in zip(bars, percs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height, f'{perc:.1f}%', ha='center', va='bottom')

        plt.xlabel('Predicate Type')
        plt.ylabel('Count (log scale)' if ylog else 'Count')
        plt.title('Distribution of First Predicates')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_image_path = f'{self.prefix}_first_predicate_distribution.jpg'
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        logging.info(f"First predicate distribution plot saved to {output_image_path}")

    def _plot_consolidated_summary(self):
        fig, axs = plt.subplots(3, 2, figsize=(20, 22))
        fig.suptitle(f'Summary for Dataset: {self.prefix}', fontsize=20)
        colors = ['skyblue', 'lightgreen', 'plum', 'aquamarine', 'gold', 'lightcoral']

        # 1. Time Distribution
        ax = axs[0, 0]
        ax.hist(self.df['runtime'], bins=20, edgecolor='black', alpha=0.7, color=colors[0])
        mean_time = self.df['runtime'].mean()
        median_time = self.df['runtime'].median()
        ax.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}s')
        ax.axvline(median_time, color='orange', linestyle='--', label=f'Median: {median_time:.2f}s')
        ax.set_title('Time Distribution')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. N-samples Distribution
        ax = axs[0, 1]
        ax.hist(self.df['n_samples'], bins=20, edgecolor='black', alpha=0.7, color=colors[1])
        mean_samples = self.df['n_samples'].mean()
        median_samples = self.df['n_samples'].median()
        ax.axvline(mean_samples, color='red', linestyle='--', label=f'Mean: {mean_samples:.2f}')
        ax.axvline(median_samples, color='orange', linestyle='--', label=f'Median: {median_samples:.2f}')
        ax.set_title('#Samples Distribution')
        ax.set_xlabel('#Samples')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Goal Distribution
        ax = axs[1, 0]
        if 'goals' in self.df.columns and not self.df['goals'].empty:
            goals = self.df['goals'].explode().dropna().tolist()
            goals = [str(g) for g in goals]
            if goals:
                counter = Counter(goals)
                sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
                labels = [item[0] for item in sorted_items]
                counts = [item[1] for item in sorted_items]
                ax.bar(labels, counts, color=colors[2])
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title('Distribution of Geometric Problem Goals')
        ax.set_xlabel('Goal Type')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)


        # 4. First Predicate Distribution
        ax = axs[1, 1]
        if 'first_predicate' in self.df.columns and not self.df['first_predicate'].empty:
            predicates = self.df['first_predicate'].dropna().tolist()
            if predicates:
                counter = Counter(predicates)
                sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
                labels = [item[0] for item in sorted_items]
                counts = [item[1] for item in sorted_items]
                ax.bar(labels, counts, color=colors[3])
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title('Distribution of First Predicates')
        ax.set_xlabel('Predicate Type')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

        # 5. Proof Length Distribution
        ax = axs[2, 0]
        if 'n_proof_steps' in self.df.columns and not self.df['n_proof_steps'].empty:
            proof_steps = self.df['n_proof_steps'].explode().dropna()
            if not proof_steps.empty:
                ax.hist(proof_steps, bins=20, edgecolor='black', alpha=0.7, color=colors[4])
                mean_len = proof_steps.mean()
                median_len = proof_steps.median()
                ax.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.2f}')
                ax.axvline(median_len, color='orange', linestyle='--', label=f'Median: {median_len:.2f}')
                ax.legend()
        ax.set_title('Proof Length Distribution')
        ax.set_xlabel('Proof Length (steps)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # 6. Clauses Num Distribution
        ax = axs[2, 1]
        if 'n_clauses' in self.df.columns and not self.df['n_clauses'].empty:
            ax.hist(self.df['n_clauses'], bins=10, edgecolor='black', alpha=0.7, color=colors[5])
            mean_clauses = self.df['n_clauses'].mean()
            median_clauses = self.df['n_clauses'].median()
            ax.axvline(mean_clauses, color='red', linestyle='--', label=f'Mean: {mean_clauses:.2f}')
            ax.axvline(median_clauses, color='orange', linestyle='--', label=f'Median: {median_clauses:.2f}')
            ax.legend()
        ax.set_title('# Clauses Distribution')
        ax.set_xlabel('# Clauses')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)



        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_image_path = f'{self.prefix}_consolidated_summary.jpg'
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        logging.info(f"Consolidated summary plot saved to {output_image_path}")

    def _convert_types_for_json(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_types_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_types_for_json(element) for element in obj]
        else:
            return obj

    def output_report(self,
                      consolidated_summary: bool = True,
                      plot_time: bool = True, 
                      plot_nsamples: bool = True, 
                      plot_goals: bool = True, 
                      plot_first_predicate: bool = True,
                      plot_proof_length: bool = True,
                      plot_clauses_num: bool = True):
        if self.df is None or self.df.empty:
            logging.warning("No data loaded, cannot generate report.")
            return

        # Unified command-line output
        print("\n--- Generation Summary ---")
        print(f"Dataset Prefix: {self.prefix}")
        
        summary_stats = {
            "Total Problems Processed": len(self.df),
            "Total Samples Generated": self.total_samples_generated,
            "Total Elapsed Time": f"{self.total_elapsed_time:.2f}s",
            "Average Runtime (s)": f"{self.df['runtime'].mean():.2f}",
            "Median Runtime (s)": f"{self.df['runtime'].median():.2f}",
        }
        if self.total_elapsed_time > 0:
            summary_stats["Average Speed (samples/s)"] = f"{self.total_samples_generated / self.total_elapsed_time:.2f}"

        if 'n_proof_steps' in self.df.columns and not self.df['n_proof_steps'].explode().empty:
            summary_stats["Average Proof Steps"] = f"{self.df['n_proof_steps'].explode().mean():.2f}"
            summary_stats["Median Proof Steps"] = f"{self.df['n_proof_steps'].explode().median():.2f}"
        if 'n_clauses' in self.df.columns:
            summary_stats["Average Clauses"] = f"{self.df['n_clauses'].mean():.2f}"
            summary_stats["Median Clauses"] = f"{self.df['n_clauses'].median():.2f}"


        for key, value in summary_stats.items():
            print(f"{key:<25}: {value}")
        
        print("------------------------\n")

        # Detailed report generation
        report_path = f'{self.prefix}_detailed_report.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Create aggregated data instead of raw_data
        aggregated_data = {}
        for column in self.df.columns:
            if column in ['goals', 'n_proof_steps']:
                # For columns with lists that need to be exploded
                exploded_series = self.df[column].explode().dropna()
                aggregated_data[f"{column}_distribution"] = exploded_series.value_counts().to_dict()
            elif self.df[column].dtype == 'object':
                # For other categorical-like data
                aggregated_data[f"{column}_distribution"] = self.df[column].value_counts().to_dict()
            else:
                # For numerical data, we can provide distribution counts as well
                aggregated_data[f"{column}_distribution"] = self.df[column].value_counts().to_dict()

        report_data = {
            "summary_statistics": summary_stats,
            **aggregated_data
        }
        
        report_data = self._convert_types_for_json(report_data)

        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            logging.info(f"Detailed report saved to {report_path}")
        except Exception as e:
            logging.error(f"Failed to save detailed report: {e}")


        if consolidated_summary:
            self._plot_consolidated_summary()
        else:
            if plot_time:
                self._plot_time_distribution()
            if plot_nsamples:
                self._plot_nsamples_distribution()
            if plot_goals:
                self._plot_goal_distribution()
            if plot_first_predicate:
                self._plot_first_predicate_distribution()
            if plot_proof_length:
                self._plot_proof_length_distribution()
            if plot_clauses_num:
                self._plot_clauses_num_distribution()