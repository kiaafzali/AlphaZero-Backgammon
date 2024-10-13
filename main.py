import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import numpy as np
import time
import logging
import sys

from bg import Backgammon
from model import ResNet
from train import train
from selfplay import selfPlay
from selfplayParallel import selfPlayParallel
from util import timed, roll_dice

# Setup logger
def setup_logger(args):
    logger = logging.getLogger('backgammon_ai')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = f"{args['dir']}/num_searches{args['num_searches']}/backgammon_ai.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# def selfPlay_wrapper(shared_model, game, args, board=None, jumps=None):
#     # This wrapper function will be called by each process
#     print(f"Starting Process {os.getpid()}")
#     return selfPlay(shared_model, game, args, board, jumps)

def selfPlayParallel_wrapper(shared_model, game, args, board=None, jumps=None, pick_idx=None):
    # This wrapper function will be called by each process
    logger = logging.getLogger('backgammon_ai')
    logger.info(f"Starting Process {os.getpid()}")
    memory= selfPlayParallel(shared_model, game, args, board, jumps)
    picke_memory(memory, args, pick_idx)
    return []

def picke_memory(memory, args, pick_idx):
    start = time.time()
    dir = args['dir']
    iter = args['train_iteration']

    output_file = f'{dir}/num_searches{args["num_searches"]}/data{iter}_{pick_idx}.pkl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'ab') as f:
        pickle.dump(memory, f)
    end = time.time()
    logger = logging.getLogger('backgammon_ai')
    logger.info(f"{len(memory)} games has been saved to {output_file} in {end - start} seconds")

def load_combined_memory(args):
    dir = args['dir']
    iter = args['train_iteration']
    combined_memory = []
    for pickle_idx in range(args["num_total_processes"]):
        input_file = f'{dir}/num_searches{args["num_searches"]}/data{iter}_{pickle_idx}.pkl'
        with open(input_file, 'rb') as f:
            memory = pickle.load(f)
            combined_memory.extend(memory)
        # delete the pickle file
        os.remove(input_file)
    picke_memory(combined_memory, args, 0)
    return combined_memory


if __name__ == "__main__":
    has_gpu = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_built()
    device =  "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    #device = "cpu"
    
    bg = Backgammon()
    shared_model = ResNet(game=bg, num_resBlocks=20, num_hidden=64, num_features=6, device=device)
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=0.001, weight_decay=0.0001)

    num_parallel_games = 512
    num_concurrent_processes = 1
    total_processes = 1

    INIT_BOARD = np.array([0, 2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0])
    TEST_BOARD = np.array([0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0])
    board = TEST_BOARD
    jumps = [1,3]

    args = {
        'dir': 'data/TEST_13',
        'num_searches': 100,
        'train_iteration': 0,
        'num_parallel_games': num_parallel_games,
        'num_concurrent_processes': num_concurrent_processes,
        'num_total_processes': total_processes,
        'num_epochs': 3,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.01,
        'C': 2, 
    }


    logger = setup_logger(args)
    logger.info(f"Starting main process {os.getpid()} with args \n{args}, \nboard = {board}, \njumps = {jumps}")
    logger.info(f"device: {device}")
    dir = args['dir']
    os.makedirs(os.path.dirname(f"{dir}/num_searches{args['num_searches']}/"), exist_ok=True)
    # torch.save(shared_model.state_dict(), f"{dir}/num_searches{args['num_searches']}/model_0.pt")
    # torch.save(optimizer.state_dict(), f"{dir}/num_searches{args['num_searches']}/optimizer_0.pt")
    shared_model.load_state_dict(torch.load(f"{dir}/num_searches{args['num_searches']}/model_1.pt"))
    optimizer.load_state_dict(torch.load(f"{dir}/num_searches{args['num_searches']}/optimizer_1.pt"))
    for i in range(1, 2):
        args['train_iteration'] = i
        shared_model.eval()
        shared_model.share_memory()  # Make the model shareable between processes

        start = time.time()
        logger.info(f"Starting data generation for iteration {i} with {num_concurrent_processes} concurrent processes and {total_processes} processes")

        # Create a pool of worker processes
        mp.set_start_method('spawn', force=True)  # Use 'spawn' method for compatibility
        with mp.Pool(processes=num_concurrent_processes) as pool:
            tasks = []
            for pick_idx in range(total_processes):
                board = INIT_BOARD if board is None else board
                jumps = roll_dice() if jumps is None else jumps
                tasks.append((shared_model, bg, args, board, jumps, pick_idx))
            
            results = []
            for result in pool.starmap(selfPlayParallel_wrapper, tasks):
                results.append(result)
                logger.info(f"  Completed process {len(results)}/{total_processes}")

        end = time.time()
        
        
        combined_memory = load_combined_memory(args)
        
        completed_games = args['num_parallel_games'] * total_processes
        states = len(combined_memory)
        
        logger.info(f"Data generation time: {end - start} seconds for {completed_games} games and {states} states")

        shared_model.train()
        logger.info(f"Starting training for iteration {i} for {states} states for {args['num_epochs']} epochs")
        start = time.time()
        train(shared_model, optimizer, bg, args, combined_memory)
        end = time.time()
        logger.info(f"Training completed in {end - start} seconds")
        
        torch.save(shared_model.state_dict(), f"{dir}/num_searches{args['num_searches']}/model_{i+1}.pt")
        torch.save(optimizer.state_dict(), f"{dir}/num_searches{args['num_searches']}/optimizer_{i+1}.pt")
        logger.info(f"Saved model and optimizer version {i+1}")
 



