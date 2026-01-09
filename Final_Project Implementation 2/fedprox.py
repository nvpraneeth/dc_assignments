"""
Centralized Federated Learning (FedProx) Implementation
Extends FedAvg with proximal regularization for non-IID robustness
"""

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import time
import os
from utils import (
    LeNet5, HybridClock, create_non_iid_dataset,
    train_local_model_prox, test_model, aggregate_models,
    get_payload_size_bits, copy_model_state, load_model_state,
    ProcessLogger, read_params_file
)


def worker_process(rank: int, num_nodes: int, r_max: int, epochs: int,
                  bandwidth_mbps: float, mu: float, train_loader: DataLoader,
                  test_loader: DataLoader, shared_models: list,
                  shared_locks: list, shared_stats: dict, log_dir: str):
    """
    Worker process for FedProx
    rank: Process ID (0 is coordinator)
    """
    device = 'cpu'
    clock = HybridClock()
    
    # Initialize logger
    logger = ProcessLogger(rank, log_dir, "FedProx (Centralized)")
    
    # Initialize model
    model = LeNet5(num_classes=10).to(device)
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
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
    logger.info(f"Proximal coefficient (mu): {mu}")
    
    for round_num in range(1, r_max + 1):
        logger.round_start(round_num, r_max, clock.get_time())
        
        # --- PHASE 1: COMPUTE ---
        logger.info("[PHASE 1] Starting local training with proximal term...")
        t_start = clock.start_compute()
        
        # Train locally with proximal regularization
        global_state = copy_model_state(model)
        train_local_model_prox(model, train_loader, epochs, global_state, mu, device)
        
        # Prepare full model for upload
        model_state = copy_model_state(model)
        
        t_end = clock.end_compute(t_start)
        compute_time = t_end
        logger.compute_phase(compute_time, epochs)
        
        # --- PHASE 2: NETWORK (Upload to coordinator) ---
        logger.info("[PHASE 2] Starting network communication...")
        payload_bits = get_payload_size_bits(model)
        network_delay = clock.add_network_delay(payload_bits, bandwidth_mbps)
        total_data_volume += payload_bits
        total_messages += 1  # Upload message
        
        if rank == 0:
            logger.network_phase("FedProx", "Receiving models from workers",
                               payload_bits/1e6, network_delay, bandwidth_mbps)
        else:
            logger.network_phase("FedProx", "Uploading to coordinator",
                               payload_bits/1e6, network_delay, bandwidth_mbps, peer=0)
        
        # Upload model to coordinator (rank 0)
        logger.synchronization("Upload", "Placing model in shared memory")
        if rank != 0:
            with shared_locks[0]:
                shared_models[rank] = model_state
            logger.debug(f"Model uploaded to coordinator (slot {rank})")
        else:
            # Coordinator collects its own model
            shared_models[0] = model_state
            logger.debug("Coordinator stored own model")
        
        # Coordinator aggregates and broadcasts
        if rank == 0:
            logger.info("[COORDINATOR] Waiting for all worker models...")
            max_wait = 5.0
            wait_start = time.time()
            while time.time() - wait_start < max_wait:
                ready_count = sum(1 for i in range(num_nodes) if shared_models[i] is not None)
                if ready_count == num_nodes:
                    logger.info(f"[COORDINATOR] All {num_nodes} models received")
                    break
                time.sleep(0.05)
            else:
                logger.warning(f"[COORDINATOR] Timeout waiting for models. Received {ready_count}/{num_nodes}")
            
            # Aggregate all models
            logger.info("[COORDINATOR] Starting model aggregation...")
            models_to_aggregate = []
            for i in range(num_nodes):
                if shared_models[i] is not None:
                    temp_model = LeNet5(num_classes=10).to(device)
                    load_model_state(temp_model, shared_models[i])
                    models_to_aggregate.append(temp_model)
            
            if len(models_to_aggregate) > 0:
                logger.aggregation(len(models_to_aggregate), "FedProx")
                aggregated = aggregate_models(models_to_aggregate)
                aggregated_state = copy_model_state(aggregated)
                logger.info("[COORDINATOR] Aggregation completed, broadcasting...")
                
                for i in range(num_nodes):
                    with shared_locks[i + num_nodes]:
                        shared_models[i + num_nodes] = aggregated_state
                logger.info(f"[COORDINATOR] Broadcasted aggregated model to {num_nodes} nodes")
                time.sleep(0.1)
        
        # Download aggregated model (all nodes including coordinator)
        logger.synchronization("Download", "Waiting for aggregated model")
        max_wait = 15.0  # Increased timeout for slower systems
        wait_start = time.time()
        aggregated_state = None
        while time.time() - wait_start < max_wait:
            with shared_locks[rank + num_nodes]:
                if shared_models[rank + num_nodes] is not None:
                    aggregated_state = shared_models[rank + num_nodes]
                    logger.debug("Aggregated model received")
                    break
            time.sleep(0.1)
        
        if aggregated_state is None:
            logger.error("Timeout waiting for aggregated model - skipping round")
            stats['rounds'].append(round_num)
            stats['virtual_times'].append(clock.get_time())
            stats['accuracies'].append(0.0)
            stats['data_volume_mb'].append(total_data_volume / (8 * 1e6))
            stats['messages'].append(total_messages)
            continue
        
        # Load aggregated model
        load_model_state(model, aggregated_state)
        logger.info("Aggregated model loaded successfully")
        
        # Download network delay
        payload_bits = get_payload_size_bits(model)
        network_delay = clock.add_network_delay(payload_bits, bandwidth_mbps)
        total_data_volume += payload_bits
        total_messages += 1  # Download message
        
        if rank == 0:
            logger.network_phase("FedProx", "Broadcasting aggregated model",
                               payload_bits/1e6, network_delay, bandwidth_mbps)
        else:
            logger.network_phase("FedProx", "Downloading from coordinator",
                               payload_bits/1e6, network_delay, bandwidth_mbps, peer=0)
        
        # Clear shared state for next round
        logger.synchronization("Cleanup", "Clearing shared memory")
        with shared_locks[rank]:
            shared_models[rank] = None
        if rank == 0:
            for i in range(num_nodes, 2 * num_nodes):
                with shared_locks[i]:
                    shared_models[i] = None
        
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
        
        # Coordinator collects global stats
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


def run_fedprox(num_nodes: int, r_max: int, epochs: int, bandwidth_mbps: float,
               mu: float = 0.1, log_dir: str = 'logs/fedprox'):
    """
    Run FedProx protocol
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create Non-IID datasets
    node_datasets, test_dataset, _ = create_non_iid_dataset(num_nodes)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Shared memory for models and statistics
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    shared_models = manager.list([None] * (2 * num_nodes))
    shared_locks = [manager.Lock() for _ in range(2 * num_nodes)]
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
            args=(rank, num_nodes, r_max, epochs, bandwidth_mbps, mu,
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
    params = read_params_file('inp-fedprox.txt')
    N = int(params[0])
    R_max = int(params[1])
    E = int(params[2])
    B = float(params[3])
    Protocol = int(params[4])
    mu = float(params[6]) if len(params) > 6 else 0.1
    
    if Protocol == 5:
        stats = run_fedprox(N, R_max, E, B, mu)
        print("FedProx completed!")
        print(f"Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        print(f"Total Virtual Time: {stats['virtual_times'][-1]:.2f}s")
    else:
        print("Set Protocol to 5 in inp-fedprox.txt to run FedProx")

