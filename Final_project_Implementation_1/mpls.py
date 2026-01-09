"""
MPLS-style Decentralized Federated Learning Implementation
Protocol 2: Smart peer selection with partial model transfer and metadata overhead
"""

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import random
from utils import (
    LeNet5, HybridClock, create_non_iid_dataset, 
    train_local_model, test_model, aggregate_models,
    get_payload_size_bits, copy_model_state, load_model_state,
    get_model_size_bits, ProcessLogger, read_params_file
)


def calculate_model_similarity(model1: LeNet5, model2: LeNet5, 
                              layer_name: str = 'fc3') -> float:
    """
    Calculate similarity between two models based on a specific layer
    Used for smart peer selection
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    # Find weights for the specified layer
    layer_key = None
    for key in state1.keys():
        if layer_name in key and 'weight' in key:
            layer_key = key
            break
    
    if layer_key is None:
        return 0.0
    
    w1 = state1[layer_key].flatten()
    w2 = state2[layer_key].flatten()
    
    # Cosine similarity
    dot_product = torch.dot(w1, w2)
    norm1 = torch.norm(w1)
    norm2 = torch.norm(w2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = (dot_product / (norm1 * norm2)).item()
    return similarity


def select_smart_peer(rank: int, num_nodes: int, model: LeNet5, 
                     candidate_set_size: int, shared_models: list,
                     shared_locks: list) -> int:
    """
    Smart peer selection: Query candidates and select best match
    Returns selected peer ID
    """
    available_peers = [i for i in range(num_nodes) if i != rank]
    
    if len(available_peers) <= candidate_set_size:
        candidates = available_peers
    else:
        candidates = random.sample(available_peers, candidate_set_size)
    
    # Query candidates (metadata messages)
    similarities = []
    for candidate_id in candidates:
        # Try to get candidate's model for similarity calculation
        candidate_model_state = None
        with shared_locks[candidate_id]:
            if shared_models[candidate_id] is not None:
                candidate_model_state = shared_models[candidate_id]
        
        if candidate_model_state is not None:
            temp_model = LeNet5(num_classes=10)
            load_model_state(temp_model, candidate_model_state)
            similarity = calculate_model_similarity(model, temp_model)
            similarities.append((candidate_id, similarity))
        else:
            similarities.append((candidate_id, 0.0))
    
    # Select peer with highest similarity (or lowest for diversity)
    # For MPLS, we might want complementary models, so select lowest similarity
    if similarities:
        selected_peer = min(similarities, key=lambda x: x[1])[0]
    else:
        selected_peer = random.choice(available_peers)
    
    return selected_peer


def worker_process(rank: int, num_nodes: int, r_max: int, epochs: int, 
                  bandwidth_mbps: float, candidate_set_size: int,
                  train_loader: DataLoader, test_loader: DataLoader, 
                  shared_models: list, shared_locks: list, 
                  shared_stats: dict, log_dir: str):
    """
    Worker process for MPLS
    """
    device = 'cpu'
    clock = HybridClock()
    random.seed(rank + 42)
    np.random.seed(rank + 42)
    
    # Initialize logger
    logger = ProcessLogger(rank, log_dir, "MPLS-style DFL")
    
    # Initialize model
    model = LeNet5(num_classes=10).to(device)
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Partial layers to transfer (e.g., last few layers)
    partial_layers = ['fc2', 'fc3']  # Transfer only fully connected layers
    logger.info(f"Partial model layers: {partial_layers}")
    logger.info(f"Candidate set size: {candidate_set_size}")
    
    # Statistics
    stats = {
        'rounds': [],
        'virtual_times': [],
        'accuracies': [],
        'data_volume_mb': [],
        'messages': []
    }
    
    total_data_volume = 0.0  # in bits
    total_messages = 0
    
    # Log initialization
    logger.initialization(num_nodes, epochs, bandwidth_mbps)
    logger.info(f"Training dataset size: {len(train_loader.dataset)} samples")
    logger.info(f"Test dataset size: {len(test_loader.dataset)} samples")
    
    for round_num in range(1, r_max + 1):
        logger.round_start(round_num, r_max, clock.get_time())
        
        # --- PHASE 1: COMPUTE ---
        logger.info("[PHASE 1] Starting local training...")
        t_start = clock.start_compute()
        
        # Train locally
        train_local_model(model, train_loader, epochs, device)
        
        # Store current model state for others to query
        model_state = copy_model_state(model)
        logger.debug("Storing model state for peer queries")
        with shared_locks[rank]:
            shared_models[rank] = model_state
        
        # Smart peer selection (includes metadata query overhead)
        logger.info(f"[PEER_SELECTION] Starting smart peer selection (candidate set: {candidate_set_size})...")
        selected_peer = select_smart_peer(rank, num_nodes, model, 
                                         candidate_set_size, shared_models, shared_locks)
        
        # Prepare partial model for exchange
        partial_state = copy_model_state(model, partial_layers)
        partial_model_bits = get_model_size_bits(model, partial_layers)
        logger.info(f"Prepared partial model ({partial_layers}): {partial_model_bits/8/1e6:.4f} MB")
        
        t_end = clock.end_compute(t_start)
        compute_time = t_end
        logger.compute_phase(compute_time, epochs)
        
        # --- PHASE 2: NETWORK ---
        logger.info("[PHASE 2] Starting network communication...")
        # Metadata query messages (2 messages per candidate: request + response)
        metadata_bits = 1024  # Assume 1KB metadata per query
        query_messages = 2 * candidate_set_size
        query_overhead = query_messages * metadata_bits
        total_messages += query_messages
        logger.info(f"[METADATA] Query overhead: {query_messages} messages ({query_overhead/8/1e3:.2f} KB)")
        
        # Partial model transfer
        payload_bits = partial_model_bits + query_overhead
        network_delay = clock.add_network_delay(payload_bits, bandwidth_mbps)
        total_data_volume += payload_bits
        total_messages += 2  # Send and receive model
        
        logger.network_phase("MPLS", "Sending partial model + metadata", 
                           payload_bits/1e6, network_delay, bandwidth_mbps, peer=selected_peer)
        logger.info(f"[NETWORK] Partial model: {partial_model_bits/8/1e6:.4f} MB, "
                   f"Metadata: {query_overhead/8/1e3:.2f} KB")
        
        # Exchange partial models with selected peer
        # Use a symmetric exchange: place model where peer can read it, and check if peer placed theirs
        logger.synchronization("Upload", f"Placing partial model for p{selected_peer}")
        
        # Place our model where the selected peer can read it
        # Use the selected peer's lock to ensure atomic write
        with shared_locks[selected_peer]:
            shared_models[rank] = partial_state
        logger.debug(f"Partial model placed in shared memory slot {rank} for p{selected_peer} to read")
        
        # Wait for peer's partial model - check if peer has placed their model in their slot
        logger.synchronization("Download", f"Waiting for partial model from p{selected_peer}")
        time.sleep(0.2)  # Give peer time to place their model
        peer_partial_state = None
        max_wait = 30.0  # Increased timeout significantly for peer-to-peer coordination
        wait_start = time.time()
        retry_count = 0
        max_retries = 150  # 30 seconds / 0.2 seconds
        
        while peer_partial_state is None and retry_count < max_retries:
            # Check if peer has placed their model in their slot
            with shared_locks[selected_peer]:
                if shared_models[selected_peer] is not None:
                    peer_partial_state = shared_models[selected_peer]
                    shared_models[selected_peer] = None  # Clear after reading
                    logger.debug(f"Received partial model from p{selected_peer}")
                    break
            
            # Also check if peer is waiting for us (they might have selected us)
            # This helps with bidirectional exchanges
            if peer_partial_state is None:
                time.sleep(0.2)
                retry_count += 1
                if retry_count % 10 == 0:  # Log every 2 seconds
                    elapsed = time.time() - wait_start
                    logger.debug(f"Still waiting for p{selected_peer}... ({elapsed:.1f}s elapsed)")
        
        if peer_partial_state is None:
            elapsed = time.time() - wait_start
            logger.error(f"Timeout waiting for partial model from p{selected_peer} after {elapsed:.1f}s - skipping round")
            # Clear our model from shared memory
            with shared_locks[rank]:
                shared_models[rank] = None
            # Still update stats to avoid index errors
            stats['rounds'].append(round_num)
            stats['virtual_times'].append(clock.get_time())
            stats['accuracies'].append(0.0)  # Use 0 as placeholder
            stats['data_volume_mb'].append(total_data_volume / (8 * 1e6))
            stats['messages'].append(total_messages)
            continue  # Skip this round if we didn't get the model
        
        # Clear our model from shared memory after successful exchange
        with shared_locks[rank]:
            shared_models[rank] = None
        
        # Aggregate partial layers with peer's model
        logger.info(f"[AGGREGATION] Aggregating partial layers ({partial_layers}) with p{selected_peer}")
        peer_model = LeNet5(num_classes=10).to(device)
        load_model_state(peer_model, copy_model_state(model))  # Start with current model
        load_model_state(peer_model, peer_partial_state, partial_layers)  # Update partial layers
        
        # Aggregate only the partial layers
        current_state = model.state_dict()
        peer_state = peer_model.state_dict()
        
        for layer_name in partial_layers:
            for key in current_state.keys():
                if layer_name in key:
                    # Weighted average
                    current_state[key] = 0.5 * current_state[key] + 0.5 * peer_state[key]
        
        model.load_state_dict(current_state)
        logger.info("Partial layer aggregation completed")
        
        # Download network delay (receiving peer's partial model)
        network_delay = clock.add_network_delay(partial_model_bits, bandwidth_mbps)
        total_data_volume += partial_model_bits
        
        logger.network_phase("MPLS", "Receiving partial model", 
                           partial_model_bits/1e6, network_delay, bandwidth_mbps, peer=selected_peer)
        
        # Test accuracy
        logger.info("[TESTING] Evaluating model on test set...")
        accuracy = test_model(model, test_loader, device)
        logger.info(f"[TESTING] Test accuracy: {accuracy:.2f}%")
        
        # Update statistics
        stats['rounds'].append(round_num)
        stats['virtual_times'].append(clock.get_time())
        stats['accuracies'].append(accuracy)
        stats['data_volume_mb'].append(total_data_volume / (8 * 1e6))
        stats['messages'].append(total_messages)
        
        logger.statistics(round_num, clock.get_time(), accuracy, 
                        total_data_volume / (8 * 1e6), total_messages)
        logger.round_end(round_num, clock.get_time(), accuracy)
        
        # Coordinator (rank 0) collects global stats
        if rank == 0:
            shared_stats['rounds'] = stats['rounds']
            shared_stats['virtual_times'] = stats['virtual_times']
            shared_stats['accuracies'] = stats['accuracies']
            shared_stats['data_volume_mb'] = stats['data_volume_mb']
            shared_stats['messages'] = stats['messages']
    
    logger.info("=" * 80)
    logger.info(f"Process {rank} completed all {r_max} rounds")
    logger.info(f"Final Statistics: Accuracy={stats['accuracies'][-1]:.2f}%, "
               f"Virtual Time={stats['virtual_times'][-1]:.2f}s, "
               f"Total Data={stats['data_volume_mb'][-1]:.2f} MB, "
               f"Total Messages={stats['messages'][-1]}")
    logger.info("=" * 80)


def run_mpls(num_nodes: int, r_max: int, epochs: int, bandwidth_mbps: float,
            candidate_set_size: int, log_dir: str = 'logs/mpls'):
    """
    Run MPLS protocol
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create Non-IID datasets
    node_datasets, test_dataset, _ = create_non_iid_dataset(num_nodes)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Shared memory for models and statistics
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    shared_models = manager.list([None] * num_nodes)
    shared_locks = [manager.Lock() for _ in range(num_nodes)]
    shared_stats = manager.dict({
        'rounds': [],
        'virtual_times': [],
        'accuracies': [],
        'data_volume_mb': [],
        'messages': []
    })
    
    # Create train loaders for each node
    train_loaders = []
    for node_dataset in node_datasets:
        train_loaders.append(DataLoader(node_dataset, batch_size=32, shuffle=True))
    
    # Spawn processes
    processes = []
    for rank in range(num_nodes):
        p = mp.Process(
            target=worker_process,
            args=(rank, num_nodes, r_max, epochs, bandwidth_mbps, candidate_set_size,
                  train_loaders[rank], test_loader, shared_models,
                  shared_locks, shared_stats, log_dir)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Return statistics
    return dict(shared_stats)


if __name__ == '__main__':
    # Read input parameters
    params = read_params_file('inp-mpls.txt')
    N = int(params[0])
    R_max = int(params[1])
    E = int(params[2])
    B = float(params[3])
    Protocol = int(params[4])
    C = int(params[5]) if len(params) > 5 else 5
    
    if Protocol == 2:
        stats = run_mpls(N, R_max, E, B, C)
        print("MPLS completed!")
        print(f"Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        print(f"Total Virtual Time: {stats['virtual_times'][-1]:.2f}s")
    else:
        print("This script is for Protocol 2 (MPLS) only")

