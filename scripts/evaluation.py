import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from newclid.agent.ddarn import DDARN
from newclid.agent.human_agent import HumanAgent
from newclid.agent.lm import LMAgent
from newclid.api import GeometricSolverBuilder


def solve_problem(args):
    """
    Process a single problem and return whether it was solved successfully along with the time taken.
    """
    problem_name, problems_path = args
    start_time = time.time()
    try:
        solver = (
            GeometricSolverBuilder(8)
            .load_problem_from_file(problems_path, problem_name)
            .with_deductive_agent(LMAgent())
            .build()
        )
        is_solved = solver.run()
        elapsed_time = time.time() - start_time
        return (problem_name, is_solved, elapsed_time) 
    except Exception as e:
        print(f"Warning: solver crashed on problem '{problem_name}' : ({type(e)}) {e}")
        elapsed_time = time.time() - start_time 
        return (problem_name, False, elapsed_time)

def run_newclid(filepath: Path, max_workers: int = 4):
    """
    Main function, read the file and execute tasks using ProcessPoolExecutor.
    
    Parameters:
        filepath (Path): The path of the file containing problem names.
        max_workers (int): The maximum number of processes in the pool, default is 4.
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

    # Use ProcessPoolExecutor to process problems concurrently
    solved_count = 0
    processed_count = 0  
    total_time = 0 
    total_real_time = time.time()   

    if max_workers == 1:
        # Single-threaded execution
        for problem_name in problem_names:
            problem_name, is_solved, elapsed_time = solve_problem((problem_name, filepath))
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
    else:
        # Multi-threaded execution using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks and collect futures
            futures = {executor.submit(solve_problem, (name, filepath)): name for name in problem_names}

            # Process completed tasks
            for future in as_completed(futures):
                problem_name = futures[future]
                problem_name, is_solved, elapsed_time = future.result()
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

    solved_percentage = (solved_count / total_problems) * 100 if total_problems > 0 else 0
    total_real_time = time.time() - total_real_time
    print(
        f"\nSuccessfully solved {solved_count}/{total_problems} problems ({solved_percentage:.2f}%). "
        f"Total time taken: {total_time:.2f}s, realtime taken: {total_real_time:.2f}s."
    )


if __name__ == "__main__":
    problems_path = Path("problems_datasets/examples.txt")
    problems_path = Path("problems_datasets/imo_ag_30.txt")
    # problems_path = Path("problems_datasets/dev.txt")
    run_newclid(problems_path, max_workers=5)  # You can adjust the value of max_workers as needed