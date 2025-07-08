import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse 
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, OrderedDict

from newclid.agent.ddarn import DDARN
from newclid.agent.human_agent import HumanAgent
# from newclid.agent.lm import LMAgent
from newclid.api import GeometricSolverBuilder


def solve_problem(args):
    """
    Process a single problem and return whether it was solved successfully along with the time taken.
    """
    problem_name, problems_path = args
    start_time = time.time()
    try:
        solver = (
            GeometricSolverBuilder(123)
            .load_problem_from_file(problems_path, problem_name)
            .build()
        )
        is_solved = solver.run()
        elapsed_time = time.time() - start_time
        
        # Get rule match time statistics
        rule_match_stats = solver.proof.matcher.get_rule_match_time_stats()
        
        return (problem_name, is_solved, elapsed_time, rule_match_stats) 
    except Exception as e:
        print(f"Warning: solver crashed on problem '{problem_name}' : ({type(e)}) {e}")
        elapsed_time = time.time() - start_time 
        return (problem_name, False, elapsed_time, {})


def run_newclid_dataset(filepath: Path, dataset_name: str, output_file, timestamp: str):
    """
    Process a single dataset and write results to output file.
    
    Parameters:
        filepath (Path): The path of the file containing problem names.
        dataset_name (str): Name of the dataset for logging purposes.
        output_file: File object to write results to.
        timestamp (str): Timestamp string for chart filenames.
    """

    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        output_file.write(f"File {filepath} not found.\n")
        return 0, 0, 0

    # Read all problem names (every other line starting from index 0)
    problem_names = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            problem_names.append(lines[i].strip())

    total_problems = len(problem_names)
    print(f"\n=== Processing {dataset_name} ===")
    print(f"Total problems to solve: {total_problems}")
    
    output_file.write(f"\n=== {dataset_name} Results ===\n")
    output_file.write(f"Dataset: {filepath}\n")
    output_file.write(f"Total problems: {total_problems}\n")
    output_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Process problems sequentially
    solved_count = 0
    processed_count = 0  
    total_time = 0 
    success_time = 0
    failed_time = 0
    total_real_time = time.time()   
    
    # Aggregate all rule match time statistics
    all_rule_match_stats = {}
    # Store rule statistics for each successful problem for chart generation (maintain original order)
    problem_rule_stats = OrderedDict()

    for problem_name in problem_names:
        problem_name, is_solved, elapsed_time, rule_match_stats = solve_problem((problem_name, filepath))
        solved_count += 1 if is_solved else 0
        processed_count += 1  
        total_time += elapsed_time 
        
        # Store current problem's rule statistics for chart generation
        problem_rule_stats[problem_name] = rule_match_stats.copy()
        
        for rule_name, match_time in rule_match_stats.items():
            if rule_name not in all_rule_match_stats:
                all_rule_match_stats[rule_name] = 0
            all_rule_match_stats[rule_name] += match_time
        
        # Track time for successful and failed problems separately
        if is_solved:
            success_time += elapsed_time
        else:
            failed_time += elapsed_time
        
        status = 'Success' if is_solved else 'Failed'
        progress_msg = (
            f"Progress: {processed_count}/{total_problems} processed, "  
            f"Solved: {solved_count}, "
            f"Current: {problem_name} "
            f"({status}), "
            f"Time: {elapsed_time:.2f}s"
        )
        print(progress_msg)
        
        # Write individual result to file
        output_file.write(f"{problem_name}: {status} ({elapsed_time:.2f}s)\n")
        output_file.flush()  # Ensure data is written immediately

    solved_percentage = (solved_count / total_problems) * 100 if total_problems > 0 else 0
    failed_count = total_problems - solved_count
    total_real_time = time.time() - total_real_time
    
    # Calculate average times
    avg_time_all = total_time / total_problems if total_problems > 0 else 0
    avg_time_success = success_time / solved_count if solved_count > 0 else 0
    avg_time_failed = failed_time / failed_count if failed_count > 0 else 0
    
    summary_msg = (
        f"\n{dataset_name} Summary: "
        f"Successfully solved {solved_count}/{total_problems} problems ({solved_percentage:.2f}%). "
        f"Total time taken: {total_time:.2f}s, realtime taken: {total_real_time:.2f}s."
    )
    print(summary_msg)
    print(f"Average time per problem (all): {avg_time_all:.2f}s")
    print(f"Average time per problem (success): {avg_time_success:.2f}s")
    print(f"Average time per problem (failed): {avg_time_failed:.2f}s")
    
    # Write summary to file
    output_file.write(f"\nSummary:\n")
    output_file.write(f"Solved: {solved_count}/{total_problems} ({solved_percentage:.2f}%)\n")
    output_file.write(f"Failed: {failed_count}/{total_problems} ({100-solved_percentage:.2f}%)\n")
    output_file.write(f"Total computation time: {total_time:.2f}s\n")
    output_file.write(f"Total real time: {total_real_time:.2f}s\n")
    output_file.write(f"Average time per problem (all): {avg_time_all:.2f}s\n")
    output_file.write(f"Average time per problem (success): {avg_time_success:.2f}s\n")
    output_file.write(f"Average time per problem (failed): {avg_time_failed:.2f}s\n")
    output_file.write(f"Total time for successful problems: {success_time:.2f}s\n")
    output_file.write(f"Total time for failed problems: {failed_time:.2f}s\n")
    
    output_file.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_file.write("=" * 50 + "\n")
    output_file.flush()
    
    # Generate rule time distribution chart
    if problem_rule_stats:
        charts_dir = Path("results") / "charts"
        charts_dir.mkdir(exist_ok=True)
        generate_rule_time_chart(problem_rule_stats, dataset_name, charts_dir, timestamp)
    
    return solved_count, total_problems, total_time, success_time, failed_time, all_rule_match_stats


def run_all_tests():
    """
    Run tests on all specified datasets and save results to a file.
    """
    # Define datasets to test
    datasets = [
        ("problems_datasets/jgex_ag_231.txt", "JGEX_AG_231"),
        ("problems_datasets/imo_ag_30.txt", "IMO_AG_30"),
        ("problems_datasets/new_benchmark_50.txt", "NEW_BENCHMARK_50"),
    ]
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"test_results_{timestamp}.txt"
    output_path = Path("results") / output_filename
    
    # Create results directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Starting comprehensive test of {len(datasets)} datasets")
    print(f"Results will be saved to: {output_path}")
    
    overall_start_time = time.time()
    total_solved = 0
    total_problems = 0
    total_computation_time = 0
    total_success_time = 0
    total_failed_time = 0
    
    # Aggregate rule match time statistics from all datasets
    overall_rule_match_stats = {}
    
    with open(output_path, "w") as output_file:
        output_file.write(f"Newclid Test Results\n")
        output_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"Command: python test_all.py\n")
        output_file.write("=" * 50 + "\n")
        
        for dataset_path, dataset_name in datasets:
            dataset_path = Path(dataset_path)
            solved, problems, comp_time, success_time, failed_time, rule_match_stats = run_newclid_dataset(dataset_path, dataset_name, output_file, timestamp)
            total_solved += solved
            total_problems += problems
            total_computation_time += comp_time
            total_success_time += success_time
            total_failed_time += failed_time
            
            # Only aggregate rule match time statistics for successful problems
            for rule_name, match_time in rule_match_stats.items():
                if rule_name not in overall_rule_match_stats:
                    overall_rule_match_stats[rule_name] = 0
                overall_rule_match_stats[rule_name] += match_time
        
        # Write overall summary
        overall_real_time = time.time() - overall_start_time
        overall_percentage = (total_solved / total_problems) * 100 if total_problems > 0 else 0
        total_failed = total_problems - total_solved
        
        # Calculate overall average times
        overall_avg_time_all = total_computation_time / total_problems if total_problems > 0 else 0
        overall_avg_time_success = total_success_time / total_solved if total_solved > 0 else 0
        overall_avg_time_failed = total_failed_time / total_failed if total_failed > 0 else 0
        
        overall_summary = f"""
OVERALL SUMMARY
===============
Total datasets tested: {len(datasets)}
Total problems: {total_problems}
Total solved: {total_solved}
Total failed: {total_failed}
Overall success rate: {overall_percentage:.2f}%
Total computation time: {total_computation_time:.2f}s
Total real time: {overall_real_time:.2f}s
Average time per problem (all): {overall_avg_time_all:.2f}s
Average time per problem (success): {overall_avg_time_success:.2f}s
Average time per problem (failed): {overall_avg_time_failed:.2f}s
Total time for successful problems: {total_success_time:.2f}s
Total time for failed problems: {total_failed_time:.2f}s
"""
        
        print(overall_summary)
        output_file.write(overall_summary)
    
    print(f"\nAll tests completed! Results saved to: {output_path}")


def generate_rule_time_chart(problem_rule_stats, dataset_name, output_dir, timestamp):
    """
    Generate a horizontal bar chart showing rule time distribution for each successfully solved problem.
    
    Parameters:
        problem_rule_stats (dict): Dictionary mapping problem names to their rule time statistics
        dataset_name (str): Name of the dataset for the chart title
        output_dir (Path): Directory to save the chart
        timestamp (str): Timestamp string for the filename
    """
    if not problem_rule_stats:
        print(f"No successful problems to generate chart for {dataset_name}")
        return
    
    # Set chart style
    plt.style.use('default')
    
    # Get all rule names and sort by total time
    all_rules = set()
    for rule_stats in problem_rule_stats.values():
        all_rules.update(rule_stats.keys())
    
    # Calculate total time for each rule
    rule_total_times = defaultdict(float)
    for rule_stats in problem_rule_stats.values():
        for rule_name, time_val in rule_stats.items():
            rule_total_times[rule_name] += time_val
    
    # Sort rules by total time
    sorted_rules = sorted(rule_total_times.items(), key=lambda x: x[1], reverse=True)
    top_rules = [rule[0] for rule in sorted_rules]
    
    # Use tab20 color scheme for consistency
    colors = plt.cm.tab20.colors
    rule_colors = {rule: colors[i % len(colors)] for i, rule in enumerate(top_rules)}
    
    # Prepare data - filter out problems with empty or zero rule stats
    problems = []
    filtered_problem_rule_stats = {}
    
    for problem, rule_stats in problem_rule_stats.items():
        total_time = sum(rule_stats.values()) if rule_stats else 0
        # Only include problems that have non-empty rule stats with positive total time
        if rule_stats and total_time > 0:
            problems.append(problem)
            filtered_problem_rule_stats[problem] = rule_stats
    
    if not problems:
        print(f"No problems with valid rule statistics for {dataset_name}")
        return
    
    # Update problem_rule_stats to use filtered data
    problem_rule_stats = filtered_problem_rule_stats
    
    # Simplify problem names: if contains /, keep only the part after the last /
    display_problems = []
    for problem in problems:
        if '/' in problem:
            display_name = problem.split('/')[-1]
        else:
            display_name = problem
        display_problems.append(display_name)
    
    # Calculate average data
    rule_avg_data = defaultdict(float)
    for rule_stats in problem_rule_stats.values():
        total_time = sum(rule_stats.values())
        if total_time > 0:
            for rule_name, time_val in rule_stats.items():
                percentage = (time_val / total_time) * 100
                rule_avg_data[rule_name] += percentage
    
    # Calculate average percentages
    for rule_name in rule_avg_data:
        rule_avg_data[rule_name] /= len(problem_rule_stats)
    
    # Create chart with optimized size ratio
    fig, ax = plt.subplots(figsize=(14, max(6, (len(problems) + 1) * 0.35)))
    
    # Create stacked bar chart for each problem
    y_positions = np.arange(len(problems) + 1)  # +1 for average
    all_display_labels = display_problems + ['average']
    
    for i, problem in enumerate(problems):
        rule_stats = problem_rule_stats[problem]
        total_time = sum(rule_stats.values())
        
        # Sort rules for current problem by time
        sorted_problem_rules = sorted(rule_stats.items(), key=lambda x: x[1], reverse=True)
        
        left_position = 0
        for rule_name, time_val in sorted_problem_rules:
            if time_val > 0:
                percentage = (time_val / total_time) * 100
                color = rule_colors[rule_name]
                
                # Draw bar
                ax.barh(i, percentage, left=left_position, 
                       color=color, align='center', height=0.85)
                
                # Add labels - lower display threshold to retain more information
                short_rule_name = rule_name.split()[0] if rule_name else rule_name
                if percentage > 8:  # Lower threshold: show rule name and percentage for >8%
                    ax.text(left_position + percentage/2, i, f'{short_rule_name}: {percentage:.1f}%',
                           va='center', ha='center', color='black', fontsize=8, weight='bold')
                elif percentage > 3:  # Show only rule name for >3%
                    ax.text(left_position + percentage/2, i, short_rule_name,
                           va='center', ha='center', color='white', fontsize=8, weight='bold')
                # No labels for areas <3%
                
                left_position += percentage
    
    # Draw average row
    avg_sorted_rules = sorted(rule_avg_data.items(), key=lambda x: x[1], reverse=True)
    left_position = 0
    for rule_name, avg_percentage in avg_sorted_rules:
        if avg_percentage > 0:
            color = rule_colors[rule_name]
            ax.barh(len(problems), avg_percentage, left=left_position,
                   color=color, align='center', height=0.85)
            
            short_rule_name = rule_name.split()[0] if rule_name else rule_name
            if avg_percentage > 8:  # Lower threshold to show more information
                ax.text(left_position + avg_percentage/2, len(problems), 
                       f'{short_rule_name}: {avg_percentage:.1f}%',
                       va='center', ha='center', color='black', fontsize=8, weight='bold')
            elif avg_percentage > 3:
                ax.text(left_position + avg_percentage/2, len(problems), short_rule_name,
                       va='center', ha='center', color='white', fontsize=8, weight='bold')
            
            left_position += avg_percentage
    
    # Set chart properties with optimized appearance
    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_display_labels, fontsize=10)
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Problems', fontsize=12)
    ax.set_title(f'Runtime of Rules - {dataset_name}', fontsize=14, weight='bold', pad=20)
    ax.set_xlim(0, 100)
    
    # Set Y-axis limits to eliminate blank space above and below
    ax.set_ylim(-0.5, len(problems) + 0.5)
    
    # Optimize grid style
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, alpha=0.6, color='gray')
    ax.set_axisbelow(True)  # Ensure grid lines are behind data
    
    # Create legend - increase number of displayed rules
    legend_data = []
    # Sort by average usage time, show top 15 most important rules (increased count)
    sorted_avg_rules = sorted(rule_avg_data.items(), key=lambda x: x[1], reverse=True)[:15]
    
    for rule_name, avg_time in sorted_avg_rules:
        if avg_time > 1.0:  # Lower threshold: show only rules with average >1%
            short_name = rule_name.split()[0] if rule_name else rule_name
            legend_data.append((short_name, rule_colors[rule_name]))
    
    if legend_data:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in legend_data]
        labels = [name for name, _ in legend_data]
        # Adjust legend layout to support more items
        ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', 
                 fontsize=9, frameon=True, framealpha=0.9, edgecolor='gray',
                 ncol=1, columnspacing=0.5)
    
    # Adjust chart layout to leave appropriate space for legend
    plt.subplots_adjust(right=0.85)
    
    # Save chart
    chart_filename = f"{dataset_name.lower()}_rule_time_distribution_{timestamp}.png"
    chart_path = output_dir / chart_filename
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Newclid evaluation on multiple datasets.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results (default: results)")
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir != "results":
        global output_path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"test_results_{timestamp}.txt"
        output_path = Path(args.output_dir) / output_filename
    
    run_all_tests()
