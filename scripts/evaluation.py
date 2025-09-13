import os
from pathlib import Path
import time
import argparse 
import ray
from rich.live import Live
from rich.table import Table

from newclid.agent.lm import LMAgent
from newclid.api import GeometricSolverBuilder

@ray.remote(num_cpus=0, num_gpus=1)
def ray_solve_problem(args):
    """
    Process a single problem and return whether it was solved successfully along with the time taken.
    """
    pid, problem_name, problems_path, model_path, decoding_size, beam_size, search_depth, timeout = args
    start_time = time.time()
    try:
        solver = (
            GeometricSolverBuilder()
            .load_problem_from_file(problems_path, problem_name, rename=True)
            .with_deductive_agent(LMAgent(model_path, decoding_size=decoding_size,beam_size=beam_size, search_depth=search_depth))
            .build()
        )
        is_solved = solver.run(timeout=timeout)
        elapsed_time = time.time() - start_time
        return (pid, problem_name, is_solved, elapsed_time) 
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Warning: solver crashed on problem '{problem_name}' : ({type(e)}) {e}")
        elapsed_time = time.time() - start_time 
        return (pid, problem_name, False, elapsed_time)

def render_table(all_tasks_info, start_time, reorder: bool):
    total_problems = len(all_tasks_info)
    solved_count = sum(status == "Success" for _, status, _ in all_tasks_info)
    processed_count = sum(status != "Pending" for _, status, _ in all_tasks_info)

    table = Table()
    table.add_column(f"Problem Names ({solved_count} Solved /{processed_count} Processed /{total_problems} Total)", justify="left", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column(f"Time ({time.time()-start_time:.2f}s)", justify="right")
    if reorder:
        priority = {"Failed": 0, "Pending": 1, "Success": 2}
        all_tasks_info = sorted(
            all_tasks_info,
            key=lambda x: priority.get(x[1], 99)  # x[1] 就是 status
        )
    for problem_name, status, elapsed_time in all_tasks_info:
        elapsed = "-" if status == "Pending" else f"{elapsed_time:.2f}"
        table.add_row(problem_name, status, elapsed)
    return table

def solve_problems(filepath: Path, modelpath: list[str], num_cpus: int, decoding_size: int, beam_size: int, search_depth: int, timeout: int = 3600):
    """
    Main function, read the file and execute tasks using Ray.
    """
    
    # Read all problem names 
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
    problem_names = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            problem_names.append(lines[i].strip())

    print(f"Total problems to solve: {len(problem_names)}")

    # Multi-threaded execution using Ray with limited concurrent tasks
    # Initialize Ray with specified number of CPUs
    if not ray.is_initialized():
        ray.init(
            # local_mode=True,
            # include_dashboard=True, dashboard_host="0.0.0.0", dashboard_port=8265,
            ignore_reinit_error=True, num_cpus=num_cpus
        )

    total_time = 0 
    start_time = time.time()
    all_tasks_info = []
    pending_tasks = []
    
    # Submit all tasks
    for i, problem_name in enumerate(problem_names):
        task = ray_solve_problem.remote((i, problem_name, filepath, modelpath, decoding_size, beam_size, search_depth, timeout))
        all_tasks_info.append((problem_name, "Pending", 0))
        pending_tasks.append(task)
    
    # Process tasks as they complete
    with Live(refresh_per_second=1) as live:
        while pending_tasks:
            # Wait for at least one task to complete
            done_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1, timeout=5)
            # Process completed tasks
            for task in done_tasks:
                pid, problem_name, is_solved, elapsed_time = ray.get(task)
                all_tasks_info[pid] = (problem_name, "Success" if is_solved else "Failed", elapsed_time)
                total_time += elapsed_time
                    
            live.update(render_table(all_tasks_info, start_time, True))
        live.update(render_table(all_tasks_info, start_time, False))
    ray.shutdown()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Newclid evaluation with configurable paths.")
    parser.add_argument("--problems_path", type=str, default="problems_datasets/dev_jgex.txt",
                        help="Path to the problems dataset file")
    parser.add_argument("--model_path", type=str, nargs='+', help="Path to the model checkpoint")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of worker processes to use")
    parser.add_argument("--decoding_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--search_depth", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout for each problem")
    args = parser.parse_args()
    
    problems_path = Path(args.problems_path)
    solve_problems(problems_path, args.model_path, num_cpus=args.max_workers, decoding_size=args.decoding_size, beam_size=args.beam_size, search_depth=args.search_depth, timeout=args.timeout)