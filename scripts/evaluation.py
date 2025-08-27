import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_LOG_LEVEL"] = "WARNING"
from pathlib import Path
import time
import argparse 
import ray

from newclid.agent.ddarn import DDARN
from newclid.agent.human_agent import HumanAgent
from newclid.agent.lm import LMAgent
from newclid.api import GeometricSolverBuilder


def solve_problem(args):
    """
    Process a single problem and return whether it was solved successfully along with the time taken.
    """
    problem_name, problems_path, model_path, decoding_size, beam_size, search_depth = args
    start_time = time.time()
    try:
        solver = (
            GeometricSolverBuilder()
            .load_problem_from_file(problems_path, problem_name, rename=True)
            .with_deductive_agent(LMAgent(model_path, decoding_size=decoding_size,beam_size=beam_size, search_depth=search_depth))
            .build()
        )
        is_solved = solver.run(timeout=3600*2)
        elapsed_time = time.time() - start_time
        return (problem_name, is_solved, elapsed_time) 
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Warning: solver crashed on problem '{problem_name}' : ({type(e)}) {e}")
        elapsed_time = time.time() - start_time 
        return (problem_name, False, elapsed_time)

@ray.remote(num_cpus=1, num_gpus=0.5)
def ray_solve_problem(args):
    """
    Ray remote function to process a single problem.
    """
    return solve_problem(args)

def run_newclid(filepath: Path, modelpath: list[Path], num_cpus: int, decoding_size: int, beam_size: int, search_depth: int):
    """
    Main function, read the file and execute tasks using Ray.
    """

    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    # Read all problem names (every other line starting from index 0)
    problem_names = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            problem_names.append(lines[i].strip())

    total_problems = len(problem_names)
    print(f"Total problems to solve: {total_problems}")

    # Use Ray to process problems concurrently
    solved_count = 0
    processed_count = 0  
    total_time = 0 
    total_real_time = time.time()   


    # Multi-threaded execution using Ray with limited concurrent tasks
    # Initialize Ray with specified number of CPUs
    if not ray.is_initialized():
        ray.init(log_to_driver=False, ignore_reinit_error=True, num_cpus=num_cpus)

    pending_tasks = {}
    completed_tasks = set()
    
    # Submit all tasks
    for i, problem_name in enumerate(problem_names):
        task = ray_solve_problem.remote((problem_name, filepath, modelpath, decoding_size, beam_size, search_depth))
        pending_tasks[task] = problem_name
    
    # Process tasks as they complete
    while pending_tasks:
        # Wait for at least one task to complete
        done_tasks, _ = ray.wait(list(pending_tasks.keys()), timeout=0)
        
        # Process completed tasks
        for task in done_tasks:
            problem_name = pending_tasks.pop(task)
            completed_tasks.add(problem_name)
            problem_name, is_solved, elapsed_time = ray.get(task)
            solved_count += 1 if is_solved else 0
            processed_count += 1  
            total_time += elapsed_time 
            print(
                f"Progress: {processed_count}/{total_problems} processed, "  
                f"Solved: {solved_count}, "
                f"Current: {problem_name} "
                f"({'Success' if is_solved else 'Failed'}), "
                f"Time: {elapsed_time:.2f}s"
            )
    ray.shutdown()

    solved_percentage = (solved_count / total_problems) * 100 if total_problems > 0 else 0
    total_real_time = time.time() - total_real_time
    print(
        f"\nSuccessfully solved {solved_count}/{total_problems} problems ({solved_percentage:.2f}%). "
        f"Total time taken: {total_time:.2f}s, realtime taken: {total_real_time:.2f}s."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Newclid evaluation with configurable paths.")
    parser.add_argument("--problems_path", type=str, default="problems_datasets/dev_jgex.txt",
                        help="Path to the problems dataset file")
    parser.add_argument("--model_path", type=str, nargs='+', help="Path to the model checkpoint")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of worker processes to use")
    parser.add_argument("--decoding_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--search_depth", type=int, default=4)
    args = parser.parse_args()
    
    problems_path = Path(args.problems_path)
    model_path = [Path(path).resolve() for path in args.model_path]
    run_newclid(problems_path, model_path, num_cpus=args.max_workers, decoding_size=args.decoding_size, beam_size=args.beam_size, search_depth=args.search_depth)