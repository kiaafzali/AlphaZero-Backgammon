import torch
import torch.multiprocessing as mp

from bg import Backgammon
from mcts import MCTS
from model import ResNet
from selfplay import selfPlayMC
from util import *


def selfPlayMC_wrapper(shared_model, game, args, board=None, jumps=None):
    # This wrapper function will be called by each process
    print(f"Process ID: {os.getpid()}: inside selfplaceMC")
    return selfPlayMC(shared_model, game, args, board, jumps)


@timed
def run_parallel_selfPlayMC(num_processes=5):
    # Set up the shared model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bg = Backgammon()
    shared_model = ResNet(game=bg, num_resBlocks=20,
                          num_hidden=64, num_features=6, device=device)
    
    shared_model.share_memory() # This is necessary for the model to be shared across processes
    shared_model.eval()
    args = {
        'C': 2,
        'num_searches': 100,
        'num_iterations': 1,
        'num_selfPlay_iterations': 128,
        'num_parallel_games': 64,
        'num_epochs': 3,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.01,
    }
    TEST_BOARD = np.array([0, 2, 0, 0, 0, 0, -3, 0, -1,
                          0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, -2, 0])
    INIT_BOARD = np.array([0, 2, 0, 0, 0, 0, -5, 0, -3, 0,
                          0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0])

    # Create a pool of worker processes
    # mp.set_start_method('spawn', force=True)  # This is important for CUDA compatibility
    pool = mp.Pool(processes=num_processes)

    # Prepare the arguments for each process
    process_args = []
    for _ in range(num_processes):
        board = INIT_BOARD
        jumps = roll_dice()
        process_args.append((shared_model, bg, args, board, jumps))

    # Run the processes and collect results
    results = pool.starmap(selfPlayMC_wrapper, process_args)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Combine the results
    combined_memory = []
    for result in results:
        # print(f"len(result): {len(result)}")
        combined_memory.extend(result)
    # print(f"len(combined_memory): {len(combined_memory)}")
    # print_cuda_info()
    return combined_memory


if __name__ == '__main__':
    # run_single_selfPlayMC()
    # This is important for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    num_runs = 1  # Number of times to run run_parallel_selfPlayMC
    output_file = 'combined_memory_results.pkl'  # File to store the results

    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}")
        combined_memory = run_parallel_selfPlayMC(1)
        print(
            f"Run {run + 1} completed. Total memory entries: {len(combined_memory)}")

        # Append the results to the file
        with open(output_file, 'ab') as f:
            pickle.dump(combined_memory, f)

        print(f"Results from run {run + 1} appended to {output_file}")

    print(f"All runs completed. Results stored in {output_file}")
