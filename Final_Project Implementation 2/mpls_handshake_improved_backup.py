"""
Improved MPLS Handshake Implementation
Key improvements:
1. Gradient-based layer selection (selects layers with highest gradient activity)
2. More comprehensive layer coverage (fc1, fc2, fc3, optionally conv layers)
3. Layer transfer verification
4. Adaptive aggregation weights
5. Better peer selection based on gradient divergence
"""

import os
import random
import time
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from mpls_common import prepare_data_loaders, init_shared_objects, create_process_logger
from utils import (
    LeNet5,
    HybridClock,
    train_local_model,
    test_model,
    get_model_size_bits,
    copy_model_state,
    load_model_state,
    read_params_file,
)


def compute_layer_gradients(model: LeNet5, train_loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Compute gradient magnitudes for each layer to identify important layers.
    Returns a dictionary mapping layer names to their gradient magnitudes.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Zero gradients
    model.zero_grad()
    
    try:
        # Get a batch for gradient computation
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient magnitudes per layer
        layer_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_magnitude = param.grad.data.norm(2).item()
                # Extract layer name (e.g., 'fc1.weight' -> 'fc1')
                layer_name = name.split('.')[0]
                if layer_name not in layer_gradients:
                    layer_gradients[layer_name] = 0.0
                layer_gradients[layer_name] += grad_magnitude
        
        model.zero_grad()
        return layer_gradients
    except Exception as e:
        # Return empty dict if gradient computation fails
        model.zero_grad()
        return {}


def select_layers_by_gradient(
    model: LeNet5, 
    train_loader: DataLoader, 
    device: str,
    min_layers: int = 3,
    max_layers: int = 5,
    include_conv: bool = False
) -> List[str]:
    """
    Select layers to transfer based on gradient magnitudes.
    Prioritizes layers with highest gradient activity.
    
    Args:
        model: The model to analyze
        train_loader: DataLoader for gradient computation
        device: Device to use
        min_layers: Minimum number of layers to include
        max_layers: Maximum number of layers to include
        include_conv: Whether to include convolutional layers
    
    Returns:
        List of layer names to transfer (e.g., ['fc1', 'fc2', 'fc3'])
    """
    layer_gradients = compute_layer_gradients(model, train_loader, device)
    
    # Define layer priority (FC layers are typically more important for aggregation)
    all_layers = ['fc1', 'fc2', 'fc3']
    if include_conv:
        all_layers = ['conv1', 'conv2'] + all_layers
    
    # Filter to only layers that exist and have gradients
    available_layers = [layer for layer in all_layers if layer in layer_gradients]
    
    if not available_layers:
        # Fallback: use default layers
        return ['fc1', 'fc2', 'fc3']
    
    # Sort by gradient magnitude (descending)
    sorted_layers = sorted(available_layers, key=lambda x: layer_gradients[x], reverse=True)
    
    # Select top layers, but ensure we have at least min_layers
    num_layers = min(max_layers, max(min_layers, len(sorted_layers)))
    selected_layers = sorted_layers[:num_layers]
    
    return selected_layers


def verify_partial_state(state_dict: Dict, expected_layers: List[str]) -> Tuple[bool, List[str]]:
    """
    Verify that a partial state dictionary contains all expected layers.
    
    Returns:
        (is_valid, missing_layers)
    """
    found_layers = set()
    for key in state_dict.keys():
        for layer_name in expected_layers:
            if layer_name in key:
                found_layers.add(layer_name)
    
    missing_layers = [layer for layer in expected_layers if layer not in found_layers]
    is_valid = len(missing_layers) == 0
    
    return is_valid, missing_layers


def calculate_gradient_divergence(model1: LeNet5, model2: LeNet5, 
                                  train_loader: DataLoader, device: str) -> Tuple[float, Dict[str, float]]:
    """
    Calculate gradient divergence between two models, per layer.
    Models with different gradients are more complementary and beneficial to aggregate.
    
    Returns:
        (total_divergence, layer_divergences) where layer_divergences maps layer names to divergence scores
    """
    try:
        # Compute gradients for both models
        grad1 = compute_layer_gradients(model1, train_loader, device)
        grad2 = compute_layer_gradients(model2, train_loader, device)
        
        # Calculate divergence per layer
        layer_divergences = {}
        total_divergence = 0.0
        common_layers = set(grad1.keys()) & set(grad2.keys())
        
        if len(common_layers) == 0:
            return 0.0, {}
        
        for layer in common_layers:
            diff = grad1[layer] - grad2[layer]
            layer_div = diff ** 2
            layer_divergences[layer] = layer_div
            total_divergence += layer_div
        
        # Normalize by number of layers
        if len(common_layers) > 0:
            total_divergence /= len(common_layers)
        
        return total_divergence, layer_divergences
    except Exception as e:
        # Fallback: use weight-based similarity as proxy
        return 0.0, {}


def select_layers_by_divergence(
    layer_divergences: Dict[str, float],
    min_layers: int = 3,
    max_layers: int = 5,
    include_conv: bool = False
) -> List[str]:
    """
    Select layers to transfer based on maximum divergence between local and peer models.
    Layers with higher divergence benefit more from aggregation.
    
    Args:
        layer_divergences: Dictionary mapping layer names to divergence scores
        min_layers: Minimum number of layers to include
        max_layers: Maximum number of layers to include
        include_conv: Whether to include convolutional layers
    
    Returns:
        List of layer names with maximum divergence
    """
    if not layer_divergences:
        # Fallback to default layers
        return ['fc1', 'fc2', 'fc3']
    
    # Filter layers based on include_conv
    available_layers = list(layer_divergences.keys())
    if not include_conv:
        available_layers = [l for l in available_layers if not l.startswith('conv')]
    
    if not available_layers:
        return ['fc1', 'fc2', 'fc3']
    
    # Sort by divergence (descending)
    sorted_layers = sorted(available_layers, key=lambda x: layer_divergences.get(x, 0.0), reverse=True)
    
    # Select top layers
    num_layers = min(max_layers, max(min_layers, len(sorted_layers)))
    selected_layers = sorted_layers[:num_layers]
    
    return selected_layers


def select_smart_peers_improved(
    rank: int, 
    num_nodes: int, 
    model: LeNet5,
    train_loader: DataLoader,
    candidate_set_size: int, 
    num_peers: int,
    shared_models: list,
    shared_locks: list,
    device: str,
    logger
) -> Tuple[List[int], Dict[int, Dict[str, float]]]:
    """
    Improved multi-peer selection using gradient divergence.
    Selects multiple peers with complementary gradients (higher divergence = better).
    
    Returns:
        (selected_peers, peer_layer_divergences) where peer_layer_divergences maps peer_id to layer divergences
    """
    available_peers = [i for i in range(num_nodes) if i != rank]
    candidates = (
        available_peers
        if len(available_peers) <= candidate_set_size
        else random.sample(available_peers, candidate_set_size)
    )
    
    peer_divergences = []  # (peer_id, total_divergence, layer_divergences)
    divergence_messages = 0  # Count messages for divergence computation
    
    logger.info(f"[DIVERGENCE_COMPUTE] Starting divergence computation for {len(candidates)} candidates")
    
    for candidate_id in candidates:
        candidate_model_state = None
        with shared_locks[candidate_id]:
            if shared_models[candidate_id] is not None:
                candidate_model_state = shared_models[candidate_id]
        
        if candidate_model_state is not None:
            try:
                temp_model = LeNet5(num_classes=10).to(device)
                load_model_state(temp_model, candidate_model_state)
                # Set to eval mode for gradient computation
                temp_model.eval()
                model.eval()
                
                # Compute divergence (counts as 2 messages: request + response)
                divergence_messages += 2
                total_div, layer_divs = calculate_gradient_divergence(model, temp_model, train_loader, device)
                
                model.train()  # Reset to train mode
                peer_divergences.append((candidate_id, total_div, layer_divs))
                logger.debug(f"[DIVERGENCE_COMPUTE] p{candidate_id}: total_div={total_div:.4f}, "
                           f"layer_divs={list(layer_divs.keys())}")
            except Exception as e:
                # Fallback to weight-based similarity if gradient computation fails
                try:
                    from decentralized_handshake import calculate_model_similarity
                    temp_model = LeNet5(num_classes=10).to(device)
                    load_model_state(temp_model, candidate_model_state)
                    similarity = calculate_model_similarity(model, temp_model)
                    # Use inverse similarity as divergence proxy
                    divergence_messages += 2
                    peer_divergences.append((candidate_id, 1.0 - abs(similarity), {}))
                    logger.debug(f"[DIVERGENCE_COMPUTE] p{candidate_id}: fallback similarity={similarity:.4f}")
                except:
                    # Last resort: random assignment
                    peer_divergences.append((candidate_id, 0.0, {}))
        else:
            # No model available, assign low divergence
            peer_divergences.append((candidate_id, 0.0, {}))
    
    # Select top N peers with highest divergence
    peer_divergences.sort(key=lambda x: x[1], reverse=True)
    num_to_select = min(num_peers, len(peer_divergences))
    selected_peers = [peer_id for peer_id, _, _ in peer_divergences[:num_to_select]]
    
    # Build layer divergence map for selected peers
    peer_layer_divergences = {}
    for peer_id, _, layer_divs in peer_divergences[:num_to_select]:
        peer_layer_divergences[peer_id] = layer_divs
    
    logger.info(f"[DIVERGENCE_COMPUTE] Completed: {divergence_messages} messages, "
               f"selected {len(selected_peers)} peers: {selected_peers}")
    
    return selected_peers, peer_layer_divergences, divergence_messages


def adaptive_aggregate(
    local_state: Dict,
    peer_state: Dict,
    partial_layers: List[str],
    local_data_size: int,
    peer_data_size: int = None
) -> Dict:
    """
    Adaptive aggregation with data-size-based weighting.
    If peer_data_size is not provided, uses equal weighting.
    
    Args:
        local_state: Local model state dictionary
        peer_state: Peer model state dictionary
        partial_layers: Layers to aggregate
        local_data_size: Size of local dataset
        peer_data_size: Size of peer dataset (optional)
    
    Returns:
        Aggregated state dictionary
    """
    aggregated_state = local_state.copy()
    
    # Calculate weights
    if peer_data_size is not None and (local_data_size + peer_data_size) > 0:
        local_weight = local_data_size / (local_data_size + peer_data_size)
        peer_weight = peer_data_size / (local_data_size + peer_data_size)
    else:
        # Equal weighting
        local_weight = 0.5
        peer_weight = 0.5
    
    # Aggregate only the specified layers
    for layer_name in partial_layers:
        for key in aggregated_state.keys():
            if layer_name in key and key in peer_state:
                aggregated_state[key] = (
                    local_weight * aggregated_state[key] + 
                    peer_weight * peer_state[key]
                )
    
    return aggregated_state


def multi_peer_aggregate(
    local_state: Dict,
    peer_states: Dict[int, Dict],
    partial_layers: List[str],
    local_data_size: int,
    peer_data_sizes: Dict[int, int] = None
) -> Dict:
    """
    Aggregate with multiple peers simultaneously.
    
    Args:
        local_state: Local model state dictionary
        peer_states: Dictionary mapping peer_id to their state dictionaries
        partial_layers: Layers to aggregate
        local_data_size: Size of local dataset
        peer_data_sizes: Dictionary mapping peer_id to their data sizes (optional)
    
    Returns:
        Aggregated state dictionary
    """
    aggregated_state = local_state.copy()
    
    # Calculate total data size
    total_size = local_data_size
    if peer_data_sizes:
        total_size += sum(peer_data_sizes.values())
    else:
        total_size += len(peer_states) * local_data_size  # Assume same size
    
    # Calculate weights
    local_weight = local_data_size / total_size if total_size > 0 else 1.0 / (len(peer_states) + 1)
    peer_weights = {}
    for peer_id in peer_states.keys():
        if peer_data_sizes and peer_id in peer_data_sizes:
            peer_weights[peer_id] = peer_data_sizes[peer_id] / total_size if total_size > 0 else 0.0
        else:
            peer_weights[peer_id] = (1.0 - local_weight) / len(peer_states)
    
    # Aggregate layers
    for layer_name in partial_layers:
        for key in aggregated_state.keys():
            if layer_name in key:
                # Start with local weight
                aggregated_value = local_weight * aggregated_state[key]
                
                # Add contributions from all peers
                for peer_id, peer_state in peer_states.items():
                    if key in peer_state:
                        weight = peer_weights.get(peer_id, 0.0)
                        aggregated_value += weight * peer_state[key]
                
                aggregated_state[key] = aggregated_value
    
    return aggregated_state


def improved_handshake_worker(
    rank: int,
    num_nodes: int,
    r_max: int,
    epochs: int,
    bandwidth_mbps: float,
    candidate_set_size: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    shared_models: list,
    shared_locks: list,
    shared_stats: dict,
    log_dir: str,
    min_layers: int = 3,
    max_layers: int = 5,
    include_conv: bool = False,
    num_peers: int = 1,
    timeout: float = 15.0,
):
    """
    Improved handshake worker with:
    - Gradient-based layer selection
    - Layer selection based on max divergence
    - Multi-peer aggregation (send/receive from multiple nodes)
    - Detailed logging for divergence computation messages
    - Timeout-based peer communication (no synchronization)
    """
    device = "cpu"
    clock = HybridClock()
    random.seed(rank + 42)
    torch.manual_seed(rank + 42)
    logger = create_process_logger(rank, log_dir, "MPLS-Handshake-Improved")
    model = LeNet5(num_classes=10).to(device)
    
    stats = {
        "rounds": [],
        "virtual_times": [],
        "accuracies": [],
        "data_volume_mb": [],
        "messages": []
    }
    total_data_volume = 0.0
    total_messages = 0
    
    # Get local dataset size for adaptive aggregation
    local_data_size = len(train_loader.dataset)
    
    logger.initialization(num_nodes, epochs, bandwidth_mbps)
    logger.info(f"Improved MPLS Handshake with multi-peer aggregation (timeout-based)")
    logger.info(f"Candidate set size: {candidate_set_size}, Number of peers per round: {num_peers}")
    logger.info(f"Min layers: {min_layers}, Max layers: {max_layers}, Include conv: {include_conv}")
    logger.info(f"Timeout: {timeout}s for peer communication")
    
    # Initial layer selection (will be updated based on divergence)
    partial_layers = ['fc1', 'fc2', 'fc3']  # Initial default
    
    for round_num in range(1, r_max + 1):
        logger.round_start(round_num, r_max, clock.get_time())
        logger.info("[PHASE 1] Training locally...")
        t_start = clock.start_compute()
        
        # Train locally
        train_local_model(model, train_loader, epochs, device)
        
        # Select layers based on gradient activity
        if round_num == 1 or round_num % 5 == 0:  # Re-select every 5 rounds
            partial_layers = select_layers_by_gradient(
                model, train_loader, device, min_layers, max_layers, include_conv
            )
            logger.info(f"[LAYER_SELECTION] Selected layers based on gradients: {partial_layers}")
        
        # Store full model state for peer queries
        local_state = copy_model_state(model)
        with shared_locks[rank]:
            shared_models[rank] = local_state
        
        # Multi-peer selection using gradient divergence
        selected_peers, peer_layer_divergences, divergence_messages = select_smart_peers_improved(
            rank, num_nodes, model, train_loader, candidate_set_size, num_peers,
            shared_models, shared_locks, device, logger
        )
        total_messages += divergence_messages
        logger.info(f"[PEER_SELECTION] Selected {len(selected_peers)} peers: {selected_peers}")
        logger.info(f"[DIVERGENCE_MSG] Divergence computation used {divergence_messages} messages")
        
        # Select layers based on maximum divergence across selected peers
        if peer_layer_divergences:
            # Aggregate layer divergences across all selected peers
            aggregated_layer_divs = {}
            for peer_id, layer_divs in peer_layer_divergences.items():
                for layer, div in layer_divs.items():
                    if layer not in aggregated_layer_divs:
                        aggregated_layer_divs[layer] = 0.0
                    aggregated_layer_divs[layer] += div
            
            # Select layers with maximum divergence
            partial_layers = select_layers_by_divergence(
                aggregated_layer_divs, min_layers, max_layers, include_conv
            )
            logger.info(f"[LAYER_SELECTION] Selected layers based on max divergence: {partial_layers}")
            logger.info(f"[LAYER_SELECTION] Layer divergences: {aggregated_layer_divs}")
        else:
            # Fallback to gradient-based selection
            if round_num == 1 or round_num % 5 == 0:
                partial_layers = select_layers_by_gradient(
                    model, train_loader, device, min_layers, max_layers, include_conv
                )
                logger.info(f"[LAYER_SELECTION] Selected layers based on gradients: {partial_layers}")
        
        # Prepare partial model with selected layers
        partial_state = copy_model_state(model, partial_layers)
        
        # Verify partial state contains expected layers
        is_valid, missing = verify_partial_state(partial_state, partial_layers)
        if not is_valid:
            logger.warning(f"[VERIFICATION] Missing layers in partial state: {missing}")
            # Fallback: use full model state
            partial_state = copy_model_state(model)
            partial_layers = ['fc1', 'fc2', 'fc3']  # Reset to default
        
        partial_bits = get_model_size_bits(model, partial_layers)
        compute_time = clock.end_compute(t_start)
        logger.compute_phase(compute_time, epochs)
        logger.info(f"[VERIFICATION] Partial state verified: {len(partial_state)} keys, "
                   f"expected layers: {partial_layers}")
        
        # Network phase - Multi-peer exchange
        metadata_bits = 1024 * candidate_set_size * 2
        payload_bits = partial_bits + metadata_bits
        
        # Send to all selected peers
        send_messages = len(selected_peers) * 2  # Send + acknowledgment per peer
        total_messages += send_messages
        total_data_volume += payload_bits * len(selected_peers)
        network_delay = clock.add_network_delay(payload_bits * len(selected_peers), bandwidth_mbps)
        
        logger.network_phase(
            "MPLS-Handshake-Improved",
            f"Sending partial model to {len(selected_peers)} peers",
            (payload_bits * len(selected_peers)) / 1e6,
            network_delay,
            bandwidth_mbps
        )
        logger.info(f"[NETWORK] Partial model size: {partial_bits/8/1e6:.4f} MB per peer, "
                   f"Total: {(payload_bits * len(selected_peers))/8/1e6:.4f} MB, "
                   f"Layers: {partial_layers}")
        logger.info(f"[NETWORK] Sending to peers: {selected_peers}, Messages: {send_messages}")
        
        # Place partial model in shared memory for all selected peers
        with shared_locks[rank]:
            shared_models[rank] = partial_state
        logger.synchronization("Upload", f"Placed verified partial model for {len(selected_peers)} peers")
        
        # Small delay to ensure model is written to shared memory
        time.sleep(0.1)
        
        # Wait for partial models from all selected peers (timeout-based, no synchronization)
        peer_states = {}
        wait_start = time.time()
        max_wait = timeout
        
        for peer_id in selected_peers:
            peer_state = None
            peer_wait_start = time.time()
            while peer_state is None and (time.time() - peer_wait_start) < max_wait:
                with shared_locks[peer_id]:
                    candidate = shared_models[peer_id]
                    if candidate is not None and isinstance(candidate, dict):
                        # Verify peer's partial state
                        peer_is_valid, peer_missing = verify_partial_state(candidate, partial_layers)
                        if peer_is_valid:
                            peer_state = candidate
                            shared_models[peer_id] = None
                            logger.debug(f"[VERIFICATION] Peer p{peer_id} partial state verified successfully")
                        else:
                            logger.warning(f"[VERIFICATION] Peer p{peer_id} partial state missing layers: {peer_missing}")
                            # Still accept it, but log the issue
                            peer_state = candidate
                            shared_models[peer_id] = None
                if peer_state is None:
                    time.sleep(0.1)
            
            if peer_state is not None:
                peer_states[peer_id] = peer_state
                logger.info(f"[RECEIVE] Received partial model from p{peer_id}")
            else:
                logger.warning(f"[RECEIVE] Timeout waiting for partial model from p{peer_id}")
        
        # Clear our model from shared memory
        with shared_locks[rank]:
            shared_models[rank] = None
        
        if not peer_states:
            logger.error(f"Failed to receive any partial models - skipping aggregation")
            stats["rounds"].append(round_num)
            stats["virtual_times"].append(clock.get_time())
            stats["accuracies"].append(0.0)
            stats["data_volume_mb"].append(total_data_volume / (8 * 1e6))
            stats["messages"].append(total_messages)
            continue
        
        # Multi-peer aggregation
        logger.info(f"[AGGREGATION] Aggregating layers {partial_layers} with {len(peer_states)} peers: {list(peer_states.keys())}")
        current_state = model.state_dict()
        
        # Use multi-peer aggregation
        aggregated_state = multi_peer_aggregate(
            current_state,
            peer_states,
            partial_layers,
            local_data_size,
            peer_data_sizes=None  # Could be passed if available
        )
        
        model.load_state_dict(aggregated_state)
        logger.info(f"Multi-peer adaptive aggregation completed with {len(peer_states)} peers")
        
        # Network delay for receiving (all peers)
        receive_bits = partial_bits * len(peer_states)
        receive_messages = len(peer_states) * 2  # Receive + acknowledgment per peer
        total_messages += receive_messages
        network_delay = clock.add_network_delay(receive_bits, bandwidth_mbps)
        total_data_volume += receive_bits
        
        logger.network_phase(
            "MPLS-Handshake-Improved",
            f"Receiving partial models from {len(peer_states)} peers",
            receive_bits / 1e6,
            network_delay,
            bandwidth_mbps
        )
        logger.info(f"[NETWORK] Received {receive_bits/8/1e6:.4f} MB total, Messages: {receive_messages}")
        
        # Test accuracy
        accuracy = test_model(model, test_loader, device)
        stats["rounds"].append(round_num)
        stats["virtual_times"].append(clock.get_time())
        stats["accuracies"].append(accuracy)
        stats["data_volume_mb"].append(total_data_volume / (8 * 1e6))
        stats["messages"].append(total_messages)
        
        logger.statistics(
            round_num,
            clock.get_time(),
            accuracy,
            total_data_volume / (8 * 1e6),
            total_messages
        )
        logger.round_end(round_num, clock.get_time(), accuracy)
        
        if rank == 0:
            shared_stats["rounds"] = stats["rounds"]
            shared_stats["virtual_times"] = stats["virtual_times"]
            shared_stats["accuracies"] = stats["accuracies"]
            shared_stats["data_volume_mb"] = stats["data_volume_mb"]
            shared_stats["messages"] = stats["messages"]
    
    logger.info("=" * 80)
    logger.info(f"Process {rank} finished all rounds; Final Accuracy={stats['accuracies'][-1]:.2f}%")
    logger.info("=" * 80)


def run_mpls_handshake_improved(
    num_nodes: int,
    r_max: int,
    epochs: int,
    bandwidth_mbps: float,
    candidate_set_size: int,
    log_dir: str = "logs/mpls_handshake_improved",
    min_layers: int = 3,
    max_layers: int = 5,
    include_conv: bool = False,
    num_peers: int = 1,
):
    """
    Run improved MPLS handshake protocol with multi-peer support (timeout-based).
    
    Args:
        num_nodes: Number of nodes
        r_max: Maximum rounds
        epochs: Local training epochs
        bandwidth_mbps: Network bandwidth
        candidate_set_size: Size of candidate set for peer selection
        log_dir: Directory for logs
        min_layers: Minimum number of layers to transfer
        max_layers: Maximum number of layers to transfer
        include_conv: Whether to include convolutional layers
        num_peers: Number of peers to aggregate with per round (multi-peer support)
        timeout: Timeout for peer communication in seconds
    """
    os.makedirs(log_dir, exist_ok=True)
    train_loaders, test_loader = prepare_data_loaders(num_nodes)
    manager, shared_models, shared_locks, shared_stats = init_shared_objects(num_nodes)
    
    processes = []
    for rank in range(num_nodes):
        p = mp.Process(
            target=improved_handshake_worker,
            args=(
                rank,
                num_nodes,
                r_max,
                epochs,
                bandwidth_mbps,
                candidate_set_size,
                train_loaders[rank],
                test_loader,
                shared_models,
                shared_locks,
                shared_stats,
                log_dir,
                min_layers,
                max_layers,
                include_conv,
                num_peers,
                15.0,  # timeout
            ),
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    return dict(shared_stats)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run improved MPLS handshake variant")
    parser.add_argument("--params", default="inp-mpls.txt", help="Input parameter file")
    parser.add_argument("--min-layers", type=int, default=3, help="Minimum layers to transfer")
    parser.add_argument("--max-layers", type=int, default=5, help="Maximum layers to transfer")
    parser.add_argument("--include-conv", action="store_true", help="Include convolutional layers")
    parser.add_argument("--num-peers", type=int, default=1, help="Number of peers to aggregate with per round")
    
    args = parser.parse_args()
    params = read_params_file(args.params)
    N = int(params[0])
    R_max = int(params[1])
    E = int(params[2])
    B = float(params[3])
    C = int(params[5]) if len(params) > 5 else 5
    
    print("Running Improved MPLS Handshake...")
    print(f"Configuration: min_layers={args.min_layers}, max_layers={args.max_layers}, "
          f"include_conv={args.include_conv}, num_peers={args.num_peers}")
    
    stats = run_mpls_handshake_improved(
        N, R_max, E, B, C,
        min_layers=args.min_layers,
        max_layers=args.max_layers,
        include_conv=args.include_conv,
        num_peers=args.num_peers
    )
    
    print("Improved MPLS Handshake completed!")
    if stats["accuracies"]:
        print(f"Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        print(f"Total Virtual Time: {stats['virtual_times'][-1]:.2f}s")
        print(f"Total Data Volume: {stats['data_volume_mb'][-1]:.2f} MB")
        print(f"Total Messages: {stats['messages'][-1]:.0f}")

