import json
import logging
import os

import numpy as np


def get_first_predicate(fl_problem: str):
    if "?" in fl_problem:
        predicates_part, _ = fl_problem.split("?", 1)
    else:
        predicates_part = fl_problem

    first_predicate = ""
    try:
        # 检查是否包含等号，如果有等号则取等号右边的部分
        if "=" in predicates_part:
            # 取第一个等号右边的部分
            predicate_part = predicates_part.split("=", 1)[1].strip()
            # 如果有分号，只取分号前面的第一个语句
            if ";" in predicate_part:
                predicate_part = predicate_part.split(";")[0].strip()
        else:
            predicate_part = predicates_part

        # 从处理后的部分提取第一个词作为谓词
        first_predicate = predicate_part.split(" ")[0].strip()
        return first_predicate
    except IndexError:
        logging.warning(f"Could not parse first predicate for: {fl_problem}")
        return None


class Statistics:
    def __init__(self, report_path: str = ""):
        self.report_path = report_path
        self.summaries = []
        self.df = None
        self.total_elapsed_time = 0
        self.total_samples_generated = 0

    def load_summaries(self, summaries: list[dict[str, any]]):
        self.summaries.extend(summaries)
        from pandas import DataFrame

        self.df = DataFrame(self.summaries)
        logging.info(self.df.describe())

    def add(self, summary: dict[str, any]):
        self.summaries.append(summary)
        # logging.info(self.df.describe())

    def _convert_types_for_json(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._convert_types_for_json(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_types_for_json(element) for element in obj]
        else:
            return obj

    def output_report(self):
        """
        Generate and output a detailed report.

        Outputs:
        - Command-line summary with key statistics
        - JSON file with detailed aggregated data
        """
        if self.df is None or self.df.empty:
            if not self.summaries:
                logging.warning("No data loaded, cannot generate report.")
                return
            from pandas import DataFrame

            self.df = DataFrame(self.summaries)

        # Unified command-line output
        print("\n--- Generation Summary ---")
        print(f"Report: {self.report_path}")

        summary_stats = {
            "Total Problems Processed": len(self.df),
            "Total Samples Generated": self.total_samples_generated,
            "Total Elapsed Time": f"{self.total_elapsed_time:.2f}s",
            "Average Runtime (s)": f"{self.df['runtime'].mean():.2f}",
            "Median Runtime (s)": f"{self.df['runtime'].median():.2f}",
        }
        if self.total_elapsed_time > 0:
            summary_stats["Average Speed (samples/s)"] = (
                f"{self.total_samples_generated / self.total_elapsed_time:.2f}"
            )

        # 添加过滤率统计
        if "n_filtered_samples" in self.df.columns:
            total_filtered = self.df["n_filtered_samples"].sum()
            total_original = self.total_samples_generated + total_filtered
            if total_original > 0:
                filter_rate = (total_filtered / total_original) * 100
                summary_stats["Total Filtered Samples"] = int(total_filtered)
                summary_stats["Filter Rate (%)"] = f"{filter_rate:.2f}"
                summary_stats["Average Filtered per Problem"] = (
                    f"{self.df['n_filtered_samples'].mean():.2f}"
                )

        if (
            "n_proof_steps" in self.df.columns
            and not self.df["n_proof_steps"].explode().empty
        ):
            summary_stats["Average Proof Steps"] = (
                f"{self.df['n_proof_steps'].explode().mean():.2f}"
            )
            summary_stats["Median Proof Steps"] = (
                f"{self.df['n_proof_steps'].explode().median():.2f}"
            )
        if "n_clauses" in self.df.columns:
            summary_stats["Average Clauses"] = (
                f"{self.df['n_clauses'].explode().mean():.2f}"
            )
            summary_stats["Median Clauses"] = (
                f"{self.df['n_clauses'].explode().median():.2f}"
            )

        for key, value in summary_stats.items():
            print(f"{key:<25}: {value}")

        print("------------------------\n")

        # Detailed report generation
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

        # Create aggregated data instead of raw_data
        aggregated_data = {}
        # Only include goals and first_predicate distributions
        for column in ["goals", "first_predicate"]:
            if column in self.df.columns:
                # For columns with lists that need to be exploded
                exploded_series = self.df[column].explode().dropna()
                counts_dict = exploded_series.value_counts().to_dict()
                aggregated_data[f"{column}_distribution"] = counts_dict

                # Add percentage distribution for goals
                if column == "goals":
                    total_goals = sum(counts_dict.values())
                    percentage_dict = {
                        k: round((v / total_goals) * 100, 2)
                        for k, v in counts_dict.items()
                    }
                    aggregated_data[f"{column}_percentage"] = percentage_dict

        report_data = {"summary_statistics": summary_stats, **aggregated_data}

        report_data = self._convert_types_for_json(report_data)

        try:
            with open(self.report_path, "w") as f:
                json.dump(report_data, f, indent=4)
            logging.info(f"Detailed report saved to {self.report_path}")
        except Exception as e:
            logging.error(f"Failed to save detailed report: {e}")
