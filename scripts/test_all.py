import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse 
from datetime import datetime

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
        return (problem_name, is_solved, elapsed_time) 
    except Exception as e:
        print(f"Warning: solver crashed on problem '{problem_name}' : ({type(e)}) {e}")
        elapsed_time = time.time() - start_time 
        return (problem_name, False, elapsed_time)


def run_newclid_dataset(filepath: Path, dataset_name: str, output_file):
    """
    Process a single dataset and write results to output file.
    
    Parameters:
        filepath (Path): The path of the file containing problem names.
        dataset_name (str): Name of the dataset for logging purposes.
        output_file: File object to write results to.
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

    for problem_name in problem_names:
        problem_name, is_solved, elapsed_time = solve_problem((problem_name, filepath))
        solved_count += 1 if is_solved else 0
        processed_count += 1  
        total_time += elapsed_time 
        
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
    
    return solved_count, total_problems, total_time, success_time, failed_time


def run_all_tests():
    """
    Run tests on all specified datasets and save results to a file.
    """
    # Define datasets to test
    datasets = [
        ("problems_datasets/jgex_ag_231.txt", "JGEX_AG_231"),
        ("problems_datasets/imo_ag_30.txt", "IMO_AG_30"),
        ("problems_datasets/new_benchmark_50.txt", "NEW_BENCHMARK_50")
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
    
    with open(output_path, "w") as output_file:
        output_file.write(f"Newclid Test Results\n")
        output_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"Command: python test_all.py\n")
        output_file.write("=" * 50 + "\n")
        
        for dataset_path, dataset_name in datasets:
            dataset_path = Path(dataset_path)
            solved, problems, comp_time, success_time, failed_time = run_newclid_dataset(dataset_path, dataset_name, output_file)
            total_solved += solved
            total_problems += problems
            total_computation_time += comp_time
            total_success_time += success_time
            total_failed_time += failed_time
        
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
