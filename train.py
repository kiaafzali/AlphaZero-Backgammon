import torch
import torch.nn.functional as F
import random
import numpy as np
import logging

def train(model, optimizer, game, args, memory):
    logger = logging.getLogger('backgammon_ai')
    random.shuffle(memory)
    encoded_memory = []
    for hist_neutral_board, hist_jumps, hist_action_probs, hist_outcome in memory:
        encoded_board, encoded_features = game.get_encoded_state(hist_neutral_board, hist_jumps)
        encoded_memory.append((encoded_board, encoded_features, hist_action_probs, hist_outcome))
    memory = encoded_memory

    for epoch in range(args['num_epochs']):
        total_policy_loss = 0
        total_value_loss = 0
        for batchIdx in range(0, len(memory), args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            if len(sample) == 0:
                continue
            board, jumps, policy_targets, value_targets = zip(*sample)
            board, jumps, policy_targets, value_targets = np.array(board), np.array(jumps), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            board = torch.tensor(board, dtype=torch.float32, device=model.device)
            jumps = torch.tensor(jumps, dtype=torch.float32, device=model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=model.device)

            out_policy, out_value = model(board, jumps)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        total_value_loss /= len(memory)
        total_policy_loss /= len(memory)
        print(f"AZ.Train(): Total policy loss: {total_policy_loss}, Total value loss: {total_value_loss}")
        logger.info(f"    Train(): epoch {epoch+1} policy loss: {total_policy_loss}, Total value loss: {total_value_loss}")