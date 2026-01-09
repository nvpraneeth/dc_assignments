"""
MPLS Handshake with Gradient Sharing and Momentum-based DFL
Key features:
1. Gradient sharing instead of weight sharing (gradient tracking approach)
2. Timeout-based approach (no synchronization)
3. Momentum-based decentralized federated learning
"""

import os
import random
import time
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
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


def compute_gradients(model: LeNet5, train_loader: DataLoader, device: str) -> Dict[str, torch.Tensor]:
    """
    Compute gradients for all model parameters.
    Returns a dictionary mapping parameter names to their gradients.
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
        
        # Extract gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.data.clone()
        
        model.zero_grad()
        return gradients
    except Exception as e:
        model.zero_grad()
        return {}


def compute_layer_gradients(model: LeNet5, train_loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Compute gradient magnitudes for each layer to identify important layers.
    Returns a dictionary mapping layer names to their gradient magnitudes.
    """
    gradients = compute_gradients(model, train_loader, device)
    
    layer_gradients = {}
    for name, grad_tensor in gradients.items():
        layer_name = name.split('.')[0]
        grad_magnitude = grad_tensor.norm(2).item()
        if layer_name not in layer_gradients:
            layer_gradients[layer_name] = 0.0
        layer_gradients[layer_name] += grad_magnitude
    
    return layer_gradients


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
    """
    layer_gradients = compute_layer_gradients(model, train_loader, device)
    
    # Define layer priority (FC layers are typically more important for aggregation)
    all_layers = ['fc1', 'fc2', 'fc3']
    if include_conv:
        all_layers = ['conv1', 'conv2'] + all_layers
    
    # Filter to only layers that exist and have gradients
    available_layers = [layer for layer in all_layers if layer in layer_gradients]
    
    if not available_layers:
        # Fallback: use default layers (respect include_conv setting)
        if include_conv:
            return ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
        else:
            return ['fc1', 'fc2', 'fc3']
    
    # Sort by gradient magnitude (descending)
    sorted_layers = sorted(available_layers, key=lambda x: layer_gradients[x], reverse=True)
    
    # Select top layers, but ensure we have at least min_layers
    num_layers = min(max_layers, max(min_layers, len(sorted_layers)))
    selected_layers = sorted_layers[:num_layers]
    
    return selected_layers


def extract_gradients_for_layers(
    gradients: Dict[str, torch.Tensor],
    partial_layers: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Extract gradients for specified layers only.
    
    Args:
        gradients: Full gradient dictionary
        partial_layers: List of layer names to extract (e.g., ['fc1', 'fc2'])
    
    Returns:
        Dictionary containing only gradients for specified layers
    """
    partial_gradients = {}
    for name, grad in gradients.items():
        for layer_name in partial_layers:
            if layer_name in name:
                partial_gradients[name] = grad.clone()
                break
    
    return partial_gradients


def verify_partial_gradients(gradients: Dict, expected_layers: List[str]) -> Tuple[bool, List[str]]:
    """
    Verify that a partial gradient dictionary contains all expected layers.
    
    Returns:
        (is_valid, missing_layers)
    """
    found_layers = set()
    for key in gradients.keys():
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
    """
    if not layer_divergences:
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
) -> Tuple[List[int], Dict[int, Dict[str, float]], int]:
    """
    Multi-peer selection using gradient divergence.
    Selects multiple peers with complementary gradients (higher divergence = better).
    
    Returns:
        (selected_peers, peer_layer_divergences, divergence_messages)
    """
    available_peers = [i for i in range(num_nodes) if i != rank]
    candidates = (
        available_peers
        if len(available_peers) <= candidate_set_size
        else random.sample(available_peers, candidate_set_size)
    )
    
    peer_divergences = []  # (peer_id, total_divergence, layer_divergences)
    divergence_messages = 0
    
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
                temp_model.eval()
                model.eval()
                
                # Compute divergence (counts as 2 messages: request + response)
                divergence_messages += 2
                total_div, layer_divs = calculate_gradient_divergence(model, temp_model, train_loader, device)
                
                model.train()
                peer_divergences.append((candidate_id, total_div, layer_divs))
                logger.debug(f"[DIVERGENCE_COMPUTE] p{candidate_id}: total_div={total_div:.4f}")
            except Exception as e:
                logger.warning(f"[DIVERGENCE_COMPUTE] Failed for p{candidate_id}: {e}")
                peer_divergences.append((candidate_id, 0.0, {}))
        else:
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


def aggregate_gradients_with_momentum(
    local_gradients: Dict[str, torch.Tensor],
    peer_gradients: Dict[int, Dict[str, torch.Tensor]],
    partial_layers: List[str],
    momentum_buffer: Dict[str, torch.Tensor],
    momentum: float = 0.9,
    learning_rate: float = 0.01,
    local_data_size: int = 1,
    peer_data_sizes: Dict[int, int] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Aggregate gradients from multiple peers using momentum-based DFL.
    
    Args:
        local_gradients: Local gradients dictionary
        peer_gradients: Dictionary mapping peer_id to their gradient dictionaries
        partial_layers: Layers to aggregate
        momentum_buffer: Previous momentum buffer
        momentum: Momentum coefficient (default 0.9)
        learning_rate: Learning rate for gradient update
        local_data_size: Size of local dataset
        peer_data_sizes: Dictionary mapping peer_id to their data sizes (optional)
    
    Returns:
        (aggregated_gradients, updated_momentum_buffer)
    """
    aggregated_gradients = {}
    updated_momentum = {}
    
    # Calculate total data size for weighting
    total_size = local_data_size
    if peer_data_sizes:
        total_size += sum(peer_data_sizes.values())
    else:
        total_size += len(peer_gradients) * local_data_size  # Assume same size
    
    # Calculate weights
    local_weight = local_data_size / total_size if total_size > 0 else 1.0 / (len(peer_gradients) + 1)
    peer_weights = {}
    for peer_id in peer_gradients.keys():
        if peer_data_sizes and peer_id in peer_data_sizes:
            peer_weights[peer_id] = peer_data_sizes[peer_id] / total_size if total_size > 0 else 0.0
        else:
            peer_weights[peer_id] = (1.0 - local_weight) / len(peer_gradients)
    
    # Aggregate gradients for each layer
    for layer_name in partial_layers:
        for param_name in local_gradients.keys():
            if layer_name in param_name:
                # Start with local gradient weighted
                aggregated_grad = local_weight * local_gradients[param_name].clone()
                
                # Add contributions from all peers
                for peer_id, peer_grad_dict in peer_gradients.items():
                    if param_name in peer_grad_dict:
                        weight = peer_weights.get(peer_id, 0.0)
                        aggregated_grad += weight * peer_grad_dict[param_name]
                
                # Apply momentum: v = momentum * v_prev + aggregated_grad
                if param_name in momentum_buffer:
                    momentum_grad = momentum * momentum_buffer[param_name] + aggregated_grad
                else:
                    momentum_grad = aggregated_grad
                
                aggregated_gradients[param_name] = aggregated_grad
                updated_momentum[param_name] = momentum_grad
    
    return aggregated_gradients, updated_momentum


def apply_gradients_to_model(
    model: LeNet5,
    gradients: Dict[str, torch.Tensor],
    learning_rate: float = 0.01
):
    """
    Apply gradients to model parameters using SGD update.
    This is equivalent to: param = param - lr * gradient
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in gradients:
                param.data -= learning_rate * gradients[name]


def apply_momentum_gradients_to_model(
    model: LeNet5,
    momentum_gradients: Dict[str, torch.Tensor],
    learning_rate: float = 0.01
):
    """
    Apply momentum-based gradients to model parameters.
    This is equivalent to: param = param - lr * momentum_gradient
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in momentum_gradients:
                param.data -= learning_rate * momentum_gradients[name]


def momentum_handshake_worker(
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
    momentum: float = 0.9,
    learning_rate: float = 0.01,
    timeout: float = 15.0,
):
    """
    Momentum-based handshake worker with gradient sharing.
    - Shares gradients instead of weights
    - Uses timeout-based approach (no synchronization)
    - Applies momentum-based DFL aggregation
    """
    device = "cpu"
    clock = HybridClock()
    random.seed(rank + 42)
    torch.manual_seed(rank + 42)
    logger = create_process_logger(rank, log_dir, "MPLS-Momentum-Gradient")
    model = LeNet5(num_classes=10).to(device)
    
    # Initialize momentum buffer
    momentum_buffer = {}
    
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
    logger.info(f"Momentum-based MPLS Handshake with gradient sharing")
    logger.info(f"Candidate set size: {candidate_set_size}, Number of peers per round: {num_peers}")
    logger.info(f"Min layers: {min_layers}, Max layers: {max_layers}, Include conv: {include_conv}")
    logger.info(f"Momentum: {momentum}, Learning rate: {learning_rate}, Timeout: {timeout}s")
    
    # Initial layer selection (include conv if enabled)
    if include_conv:
        partial_layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
    else:
        partial_layers = ['fc1', 'fc2', 'fc3']
    
    for round_num in range(1, r_max + 1):
        logger.round_start(round_num, r_max, clock.get_time())
        logger.info("[PHASE 1] Training locally...")
        t_start = clock.start_compute()
        
        # Train locally
        train_local_model(model, train_loader, epochs, device)
        
        # Compute gradients after training (on a fresh batch for gradient tracking)
        # These gradients represent the direction the model wants to move
        local_gradients = compute_gradients(model, train_loader, device)
        if not local_gradients:
            logger.warning("[GRADIENT] Failed to compute gradients, using zero gradients")
            # Create zero gradients as fallback
            local_gradients = {}
            for name, param in model.named_parameters():
                local_gradients[name] = torch.zeros_like(param.data)
        logger.info(f"[GRADIENT] Computed gradients for {len(local_gradients)} parameters")
        
        # Select layers based on gradient activity
        if round_num == 1 or round_num % 5 == 0:
            partial_layers = select_layers_by_gradient(
                model, train_loader, device, min_layers, max_layers, include_conv
            )
            logger.info(f"[LAYER_SELECTION] Selected layers based on gradients: {partial_layers}")
        
        # Store full model state for peer queries (for divergence computation)
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
        
        # Select layers based on maximum divergence across selected peers
        if peer_layer_divergences:
            aggregated_layer_divs = {}
            for peer_id, layer_divs in peer_layer_divergences.items():
                for layer, div in layer_divs.items():
                    if layer not in aggregated_layer_divs:
                        aggregated_layer_divs[layer] = 0.0
                    aggregated_layer_divs[layer] += div
            
            partial_layers = select_layers_by_divergence(
                aggregated_layer_divs, min_layers, max_layers, include_conv
            )
            logger.info(f"[LAYER_SELECTION] Selected layers based on max divergence: {partial_layers}")
        else:
            if round_num == 1 or round_num % 5 == 0:
                partial_layers = select_layers_by_gradient(
                    model, train_loader, device, min_layers, max_layers, include_conv
                )
                logger.info(f"[LAYER_SELECTION] Selected layers based on gradients: {partial_layers}")
        
        # Extract partial gradients for selected layers
        partial_gradients = extract_gradients_for_layers(local_gradients, partial_layers)
        
        # Verify partial gradients
        is_valid, missing = verify_partial_gradients(partial_gradients, partial_layers)
        if not is_valid:
            logger.warning(f"[VERIFICATION] Missing gradients for layers: {missing}")
            # Fallback: use all gradients for selected layers
            partial_gradients = local_gradients.copy()
            # Keep the selected layers (may include conv if include_conv=True)
            if include_conv:
                partial_layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
            else:
                partial_layers = ['fc1', 'fc2', 'fc3']
        
        # Calculate gradient size in bits
        gradient_bits = 0
        for grad in partial_gradients.values():
            gradient_bits += grad.numel() * 32  # 32 bits per float32
        
        compute_time = clock.end_compute(t_start)
        logger.compute_phase(compute_time, epochs)
        logger.info(f"[VERIFICATION] Partial gradients verified: {len(partial_gradients)} parameters, "
                   f"expected layers: {partial_layers}, size: {gradient_bits/8/1e6:.4f} MB")
        
        # Network phase - Multi-peer exchange
        metadata_bits = 1024 * candidate_set_size * 2
        payload_bits = gradient_bits + metadata_bits
        
        # Send to all selected peers
        send_messages = len(selected_peers) * 2  # Send + acknowledgment per peer
        total_messages += send_messages
        total_data_volume += payload_bits * len(selected_peers)
        network_delay = clock.add_network_delay(payload_bits * len(selected_peers), bandwidth_mbps)
        
        logger.network_phase(
            "MPLS-Momentum-Gradient",
            f"Sending partial gradients to {len(selected_peers)} peers",
            (payload_bits * len(selected_peers)) / 1e6,
            network_delay,
            bandwidth_mbps
        )
        logger.info(f"[NETWORK] Partial gradient size: {gradient_bits/8/1e6:.4f} MB per peer, "
                   f"Total: {(payload_bits * len(selected_peers))/8/1e6:.4f} MB, "
                   f"Layers: {partial_layers}")
        
        # Place partial gradients in shared memory for all selected peers
        with shared_locks[rank]:
            shared_models[rank] = partial_gradients
        logger.synchronization("Upload", f"Placed partial gradients for {len(selected_peers)} peers")
        
        # Wait for partial gradients from all selected peers (timeout-based, no synchronization)
        peer_gradients = {}
        wait_start = time.time()
        max_wait = timeout
        
        for peer_id in selected_peers:
            peer_grad = None
            peer_wait_start = time.time()
            while peer_grad is None and (time.time() - peer_wait_start) < max_wait:
                with shared_locks[peer_id]:
                    candidate = shared_models[peer_id]
                    if candidate is not None and isinstance(candidate, dict):
                        # Verify peer's partial gradients
                        peer_is_valid, peer_missing = verify_partial_gradients(candidate, partial_layers)
                        if peer_is_valid:
                            peer_grad = candidate
                            shared_models[peer_id] = None
                            logger.debug(f"[VERIFICATION] Peer p{peer_id} partial gradients verified")
                        else:
                            # Still accept it, but log the issue
                            peer_grad = candidate
                            shared_models[peer_id] = None
                            logger.warning(f"[VERIFICATION] Peer p{peer_id} missing gradients: {peer_missing}")
                if peer_grad is None:
                    time.sleep(0.1)
            
            if peer_grad is not None:
                peer_gradients[peer_id] = peer_grad
                logger.info(f"[RECEIVE] Received partial gradients from p{peer_id}")
            else:
                logger.warning(f"[RECEIVE] Timeout waiting for partial gradients from p{peer_id}")
        
        # Clear our gradients from shared memory
        with shared_locks[rank]:
            shared_models[rank] = None
        
        if not peer_gradients:
            logger.error(f"Failed to receive any partial gradients - skipping aggregation")
            stats["rounds"].append(round_num)
            stats["virtual_times"].append(clock.get_time())
            stats["accuracies"].append(0.0)
            stats["data_volume_mb"].append(total_data_volume / (8 * 1e6))
            stats["messages"].append(total_messages)
            continue
        
        # Momentum-based aggregation
        logger.info(f"[AGGREGATION] Aggregating gradients for layers {partial_layers} with {len(peer_gradients)} peers using momentum")
        aggregated_grads, updated_momentum = aggregate_gradients_with_momentum(
            local_gradients,
            peer_gradients,
            partial_layers,
            momentum_buffer,
            momentum=momentum,
            learning_rate=learning_rate,
            local_data_size=local_data_size,
            peer_data_sizes=None
        )
        
        # Update momentum buffer
        momentum_buffer = updated_momentum
        
        # Apply momentum-based gradients to model
        apply_momentum_gradients_to_model(model, updated_momentum, learning_rate)
        logger.info(f"Momentum-based gradient aggregation and update completed with {len(peer_gradients)} peers")
        
        # Network delay for receiving (all peers)
        receive_bits = gradient_bits * len(peer_gradients)
        receive_messages = len(peer_gradients) * 2
        total_messages += receive_messages
        network_delay = clock.add_network_delay(receive_bits, bandwidth_mbps)
        total_data_volume += receive_bits
        
        logger.network_phase(
            "MPLS-Momentum-Gradient",
            f"Receiving partial gradients from {len(peer_gradients)} peers",
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


def run_mpls_handshake_momentum(
    num_nodes: int,
    r_max: int,
    epochs: int,
    bandwidth_mbps: float,
    candidate_set_size: int,
    log_dir: str = "logs/mpls_handshake_momentum",
    min_layers: int = 3,
    max_layers: int = 5,
    include_conv: bool = False,
    num_peers: int = 1,
    momentum: float = 0.9,
    learning_rate: float = 0.01,
    timeout: float = 15.0,
):
    """
    Run momentum-based MPLS handshake protocol with gradient sharing.
    
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
        num_peers: Number of peers to aggregate with per round
        momentum: Momentum coefficient for gradient aggregation
        learning_rate: Learning rate for gradient updates
        timeout: Timeout for waiting for peer gradients (seconds)
    """
    os.makedirs(log_dir, exist_ok=True)
    train_loaders, test_loader = prepare_data_loaders(num_nodes)
    manager, shared_models, shared_locks, shared_stats = init_shared_objects(num_nodes)
    
    processes = []
    
    try:
        for rank in range(num_nodes):
            p = mp.Process(
                target=momentum_handshake_worker,
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
                    momentum,
                    learning_rate,
                    timeout,
                )
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        print(f"\n{'='*80}")
        print(f"Momentum-based MPLS Handshake with Gradient Sharing completed!")
        print(f"Logs saved to: {log_dir}")
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MPLS Handshake with Gradient Sharing and Momentum")
    parser.add_argument("--params", type=str, default="inp-params.txt", 
                       help="Path to parameters file (default: inp-params.txt)")
    parser.add_argument("--num_peers", type=int, default=1,
                       help="Number of peers to aggregate with per round (default: 1)")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="Momentum coefficient (default: 0.9)")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                       help="Learning rate (default: 0.01)")
    parser.add_argument("--timeout", type=float, default=15.0,
                       help="Timeout for peer communication in seconds (default: 15.0)")
    parser.add_argument("--include-conv", action="store_true",
                       help="Include convolutional layers (conv1, conv2) in gradient sharing")
    parser.add_argument("--min-layers", type=int, default=3,
                       help="Minimum number of layers to share (default: 3)")
    parser.add_argument("--max-layers", type=int, default=5,
                       help="Maximum number of layers to share (default: 5)")
    
    args = parser.parse_args()
    
    try:
        params = read_params_file(args.params)
    except Exception as e:
        print(f"Error reading parameters file {args.params}: {e}")
        exit(1)
    
    if len(params) < 6:
        print(f"Error: Expected at least 6 parameters, got {len(params)}")
        exit(1)
    
    # Parse parameters (handle both 6 and 7 value formats)
    num_nodes = int(params[0])
    r_max = int(params[1])
    epochs = int(params[2])
    bandwidth_mbps = float(params[3])
    protocol = int(params[4])
    candidate_set_size = int(params[5])
    # params[6] is MU (FedProx coefficient) - not used in this script
    
    if protocol != 2:
        print("This script is for MPLS protocol (protocol=2)")
        exit(1)
    
    run_mpls_handshake_momentum(
        num_nodes=num_nodes,
        r_max=r_max,
        epochs=epochs,
        bandwidth_mbps=bandwidth_mbps,
        candidate_set_size=candidate_set_size,
        min_layers=args.min_layers,
        max_layers=args.max_layers,
        include_conv=args.include_conv,
        num_peers=args.num_peers,
        momentum=args.momentum,
        learning_rate=args.learning_rate,
        timeout=args.timeout,
    )

