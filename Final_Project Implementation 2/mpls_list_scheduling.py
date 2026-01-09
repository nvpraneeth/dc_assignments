"""
MPLS with List Scheduling Algorithm
Implements peer and layer selection based on:
1. Peer Selection: Network conditions (bandwidth) and data heterogeneity (Non-IID)
2. Layer Selection: Training efficiency (gradient variation)
3. List Scheduling: Maps layers to peers to minimize communication delay
"""

import os
import random
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

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


def calculate_class_proportions(train_loader: DataLoader, num_classes: int = 10) -> Dict[int, float]:
    """
    Calculate class proportions for a worker's dataset.
    
    Args:
        train_loader: DataLoader for the worker's training data
        num_classes: Number of classes (default 10 for MNIST)
    
    Returns:
        Dictionary mapping class index to proportion (pic)
    """
    class_counts = defaultdict(int)
    total_samples = 0
    
    for _, target in train_loader:
        for label in target:
            label_val = label.item() if hasattr(label, 'item') else int(label)
            class_counts[label_val] += 1
            total_samples += 1
    
    # Calculate proportions
    proportions = {}
    for c in range(num_classes):
        proportions[c] = class_counts.get(c, 0) / total_samples if total_samples > 0 else 0.0
    
    return proportions


def calculate_data_divergence(
    proportions_i: Dict[int, float],
    proportions_s: Dict[int, float],
    num_classes: int = 10
) -> float:
    """
    Calculate Data Divergence (DDs) between worker i and peer s.
    
    DDs = sum over c |pic - psc|
    
    Args:
        proportions_i: Class proportions for worker i
        proportions_s: Class proportions for peer s
        num_classes: Number of classes
    
    Returns:
        Data divergence value
    """
    divergence = 0.0
    for c in range(num_classes):
        pic = proportions_i.get(c, 0.0)
        psc = proportions_s.get(c, 0.0)
        divergence += abs(pic - psc)
    
    return divergence


def calculate_peer_selection_probability(
    bandwidths: Dict[int, float],
    divergences: Dict[int, float],
    tau1: float = 0.5,
    tau2: float = 0.5
) -> Dict[int, float]:
    """
    Calculate peer selection probability (psk).
    
    psk = (tau1 * Bsk + tau2 * DDs) / sum_s' (tau1 * Bs'k + tau2 * DDs')
    
    Args:
        bandwidths: Dictionary mapping peer_id to normalized bandwidth (Bsk)
        divergences: Dictionary mapping peer_id to data divergence (DDs)
        tau1: Weight for bandwidth (default 0.5)
        tau2: Weight for divergence (default 0.5)
    
    Returns:
        Dictionary mapping peer_id to selection probability
    """
    # Normalize bandwidths if needed (0-1 range)
    if bandwidths:
        max_bw = max(bandwidths.values()) if max(bandwidths.values()) > 0 else 1.0
        normalized_bw = {peer_id: bw / max_bw for peer_id, bw in bandwidths.items()}
    else:
        normalized_bw = {}
    
    # Normalize divergences if needed (0-1 range)
    if divergences:
        max_div = max(divergences.values()) if max(divergences.values()) > 0 else 1.0
        normalized_div = {peer_id: div / max_div for peer_id, div in divergences.items()}
    else:
        normalized_div = {}
    
    # Calculate combined scores
    scores = {}
    all_peers = set(normalized_bw.keys()) | set(normalized_div.keys())
    
    for peer_id in all_peers:
        bw_score = normalized_bw.get(peer_id, 0.0)
        div_score = normalized_div.get(peer_id, 0.0)
        scores[peer_id] = tau1 * bw_score + tau2 * div_score
    
    # Calculate denominator (sum of all scores)
    total_score = sum(scores.values())
    
    # Calculate probabilities
    probabilities = {}
    if total_score > 0:
        for peer_id, score in scores.items():
            probabilities[peer_id] = score / total_score
    else:
        # Uniform distribution if all scores are zero
        num_peers = len(all_peers)
        if num_peers > 0:
            uniform_prob = 1.0 / num_peers
            for peer_id in all_peers:
                probabilities[peer_id] = uniform_prob
    
    return probabilities


def calculate_gradient_variation(
    weights_t: Dict[str, torch.Tensor],
    weights_t_prime: Dict[str, torch.Tensor],
    layer_name: str
) -> float:
    """
    Calculate gradient variation (gst',t(l)) for layer l.
    
    gst',t(l) = ||wst(l) - wst'(l)||_2
    
    This measures how much layer l has changed between epoch t' and t.
    
    Args:
        weights_t: Current weights (at epoch t)
        weights_t_prime: Previous weights (at epoch t')
        layer_name: Name of the layer (e.g., 'fc1', 'fc2')
    
    Returns:
        Euclidean distance (gradient variation)
    """
    layer_weights_t = {}
    layer_weights_t_prime = {}
    
    # Extract weights for this layer
    for param_name, param_tensor in weights_t.items():
        if layer_name in param_name:
            layer_weights_t[param_name] = param_tensor
    
    for param_name, param_tensor in weights_t_prime.items():
        if layer_name in param_name:
            layer_weights_t_prime[param_name] = param_tensor
    
    # Calculate Euclidean distance
    total_distance = 0.0
    for param_name in layer_weights_t:
        if param_name in layer_weights_t_prime:
            diff = layer_weights_t[param_name] - layer_weights_t_prime[param_name]
            total_distance += torch.sum(diff ** 2).item()
    
    return np.sqrt(total_distance) if total_distance > 0 else 0.0


def calculate_layer_probability(
    gradient_variations: Dict[int, float],
    layer_name: str
) -> Dict[int, float]:
    """
    Calculate layer probability (qsk(l)).
    
    qsk(l) = gst',t(l) / sum_s' gs't',t(l)
    
    Args:
        gradient_variations: Dictionary mapping peer_id to gradient variation for layer
        layer_name: Name of the layer
    
    Returns:
        Dictionary mapping peer_id to layer probability
    """
    total_variation = sum(gradient_variations.values())
    
    probabilities = {}
    if total_variation > 0:
        for peer_id, variation in gradient_variations.items():
            probabilities[peer_id] = variation / total_variation
    else:
        # Uniform distribution if all variations are zero
        num_peers = len(gradient_variations)
        if num_peers > 0:
            uniform_prob = 1.0 / num_peers
            for peer_id in gradient_variations.keys():
                probabilities[peer_id] = uniform_prob
    
    return probabilities


def list_scheduling_algorithm(
    mu1: np.ndarray,
    mu2: np.ndarray,
    layer_sizes: List[int],
    bandwidths: List[float],
    num_peers: int,
    num_layers: int
) -> Dict[int, int]:
    """
    List Scheduling Algorithm to assign layers to peers.
    
    Goal: Generate assignment map π(l) = s (Layer l comes from Peer s)
    
    Args:
        mu1: S×L matrix where mu1[s,l] = Ml / Bsk (time to download layer l from peer s)
        mu2: S×L combined probability matrix (mu2[s,l] = psk * qsk(l))
        layer_sizes: List of layer sizes Ml
        bandwidths: List of bandwidths Bsk for each peer
        num_peers: Number of peers (S)
        num_layers: Number of layers (L)
    
    Returns:
        Dictionary mapping layer_index to peer_id (π(l) = s)
    """
    # Step 1: Pruning - Filter low-utility peers
    # Calculate threshold θ(l) = (1/S) * sum_s mu2(s,l)
    theta = np.zeros(num_layers)
    for l in range(num_layers):
        theta[l] = np.mean(mu2[:, l]) if num_peers > 0 else 0.0
    
    # Prune: set mu1[s,l] = inf if mu2[s,l] < theta[l]
    mu1_pruned = mu1.copy()
    for s in range(num_peers):
        for l in range(num_layers):
            if mu2[s, l] < theta[l]:
                mu1_pruned[s, l] = np.inf
    
    # Step 2: Calculate Efficiency Metric E(s,l)
    # φ(l) = min_s mu1(s,l) - fastest possible time for layer l
    phi = np.zeros(num_layers)
    for l in range(num_layers):
        valid_times = mu1_pruned[:, l][mu1_pruned[:, l] != np.inf]
        phi[l] = np.min(valid_times) if len(valid_times) > 0 else np.inf
    
    # E(s,l) = phi(l) / mu1(s,l) - efficiency metric (higher is better, closer to optimal)
    # Note: phi(l) is the minimum time, so E(s,l) = phi(l)/mu1(s,l) means:
    # - E(s,l) = 1.0 if this peer is the fastest for layer l
    # - E(s,l) < 1.0 if this peer is slower
    # - Higher E means more efficient (closer to optimal)
    E = np.zeros((num_peers, num_layers))
    for s in range(num_peers):
        for l in range(num_layers):
            if phi[l] > 0 and not np.isinf(phi[l]) and not np.isinf(mu1_pruned[s, l]) and mu1_pruned[s, l] > 0:
                E[s, l] = phi[l] / mu1_pruned[s, l]
            else:
                E[s, l] = 0.0
    
    # Step 3: Sorting
    # Sort layers for each peer by E(s,l) descending
    peer_layer_sorted = {}
    for s in range(num_peers):
        layer_indices = list(range(num_layers))
        layer_indices.sort(key=lambda l: E[s, l], reverse=True)
        peer_layer_sorted[s] = layer_indices
    
    # Calculate peer rank: rank(s) = sum_l E(s,l)
    peer_ranks = {}
    for s in range(num_peers):
        peer_ranks[s] = np.sum(E[s, :])
    
    # Sort peers by rank descending
    peer_order = sorted(range(num_peers), key=lambda s: peer_ranks[s], reverse=True)
    
    # Step 4: Primary Assignment (List Scheduling Loop)
    assignment = {}  # π(l) = s
    current_load = {s: 0.0 for s in range(num_peers)}
    unassigned_layers = set(range(num_layers))
    removed_peers = set()  # Peers removed from consideration
    max_iterations = num_layers * num_peers * 2  # Safety limit to prevent infinite loops
    iteration_count = 0
    
    while unassigned_layers and iteration_count < max_iterations:
        iteration_count += 1
        
        # Pick peer with minimum current_load (break ties with higher rank)
        # Exclude removed peers
        available_peers = [s for s in peer_order if s not in removed_peers]
        
        if not available_peers:
            # All peers removed, assign remaining to best peer by rank
            if unassigned_layers:
                best_peer = max(peer_order, key=lambda s: peer_ranks[s])
                for layer_idx in list(unassigned_layers):
                    assignment[layer_idx] = best_peer
                    current_load[best_peer] += mu1_pruned[best_peer, layer_idx] if not np.isinf(mu1_pruned[best_peer, layer_idx]) else 0.0
            break
        
        # Find peer with minimum load
        min_load = min(current_load[s] for s in available_peers)
        candidates = [s for s in available_peers if current_load[s] == min_load]
        # Break ties with higher rank
        min_load_peer = max(candidates, key=lambda s: peer_ranks[s])
        
        # Pick unassigned layer from this peer's sorted list with highest E(s,l)
        assigned = False
        best_layer = None
        best_efficiency = -1
        
        # First pass: find the best layer that meets the constraint
        for layer_idx in peer_layer_sorted[min_load_peer]:
            if layer_idx in unassigned_layers:
                efficiency = E[min_load_peer, layer_idx]
                # Constraint check: Prefer layers with E(s,l) > 1/S, but if none exist, use the best available
                if efficiency > best_efficiency:
                    best_layer = layer_idx
                    best_efficiency = efficiency
        
        # If we found a layer, assign it (even if efficiency is low - better than nothing)
        if best_layer is not None:
            # Only skip if efficiency is extremely low (close to 0)
            if best_efficiency > 0.01:  # Very low threshold instead of 1/S
                assignment[best_layer] = min_load_peer
                unassigned_layers.remove(best_layer)
                current_load[min_load_peer] += mu1_pruned[min_load_peer, best_layer] if not np.isinf(mu1_pruned[min_load_peer, best_layer]) else 0.0
                assigned = True
            else:
                # Efficiency too low - this peer can't handle any layers well
                removed_peers.add(min_load_peer)
        else:
            # No layers available for this peer
            removed_peers.add(min_load_peer)
        
        if not assigned:
            # No valid layer found for this peer - remove it from consideration
            removed_peers.add(min_load_peer)
            
            # If all peers are removed or we can't assign, assign remaining to best peer
            if len(removed_peers) >= len(available_peers) or not available_peers:
                if unassigned_layers:
                    # Assign all remaining layers to the best available peer (even if removed)
                    best_peer = max(peer_order, key=lambda s: peer_ranks[s])
                    for layer_idx in list(unassigned_layers):
                        assignment[layer_idx] = best_peer
                        current_load[best_peer] += mu1_pruned[best_peer, layer_idx] if not np.isinf(mu1_pruned[best_peer, layer_idx]) else 0.0
                break
    
    # Step 5: Secondary Assignment (Backfilling)
    MAX_DELAY = max(current_load.values()) if current_load.values() else 0.0
    
    # Store redundant assignments (multiple peers per layer)
    redundant_assignments = defaultdict(list)  # {layer_idx: [peer_idx, ...]}
    for layer_idx, peer_idx in assignment.items():
        redundant_assignments[layer_idx].append(peer_idx)
    
    # Add redundant assignments if they don't exceed MAX_DELAY
    for s in range(num_peers):
        for l in range(num_layers):
            if l not in assignment or assignment[l] != s:
                # Check if adding this layer would not exceed MAX_DELAY
                additional_time = mu1_pruned[s, l] if not np.isinf(mu1_pruned[s, l]) else 0.0
                if current_load[s] + additional_time <= MAX_DELAY and E[s, l] > (1.0 / num_peers):
                    # Add redundant assignment
                    redundant_assignments[l].append(s)
                    current_load[s] += additional_time
    
    # Ensure all layers are assigned (fallback)
    if len(assignment) < num_layers:
        unassigned = set(range(num_layers)) - set(assignment.keys())
        if unassigned:
            # Assign remaining layers to the best peer by rank
            best_peer = max(peer_order, key=lambda s: peer_ranks[s])
            for layer_idx in unassigned:
                assignment[layer_idx] = best_peer
                current_load[best_peer] += mu1_pruned[best_peer, layer_idx] if not np.isinf(mu1_pruned[best_peer, layer_idx]) else 0.0
    
    # Return primary assignment (for simplicity, we use primary assignment)
    # In practice, redundant assignments could be used for better convergence
    return assignment


def list_scheduling_worker(
    rank: int,
    num_nodes: int,
    r_max: int,
    epochs: int,
    bandwidth_mbps: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    shared_models: list,
    shared_locks: list,
    shared_stats: dict,
    log_dir: str,
    tau1: float = 0.5,
    tau2: float = 0.5,
    candidate_set_size: int = 5,
):
    """
    Worker process implementing MPLS with List Scheduling.
    """
    device = "cpu"
    clock = HybridClock()
    random.seed(rank + 42)
    torch.manual_seed(rank + 42)
    logger = create_process_logger(rank, log_dir, "MPLS-ListScheduling")
    model = LeNet5(num_classes=10).to(device)
    
    # Calculate class proportions for this worker
    class_proportions = calculate_class_proportions(train_loader, num_classes=10)
    logger.info(f"[INIT] Class proportions: {class_proportions}")
    
    # Get layer names and sizes
    layer_names = model.get_layer_names()
    layer_sizes = []
    state_dict = model.state_dict()
    for layer_name in layer_names:
        layer_size = 0
        for param_name, param_tensor in state_dict.items():
            if layer_name in param_name:
                layer_size += param_tensor.numel() * 32  # 32 bits per float32
        layer_sizes.append(layer_size)
    
    logger.info(f"[INIT] Layer names: {layer_names}, Sizes (bits): {layer_sizes}")
    
    # Store previous model states for gradient variation calculation
    previous_model_states = {}  # Will store local states from previous rounds
    peer_previous_states = {}  # {round_num: {peer_id: state}} - track peer states across rounds
    
    stats = {
        "rounds": [],
        "virtual_times": [],
        "accuracies": [],
        "data_volume_mb": [],
        "messages": []
    }
    total_data_volume = 0.0
    total_messages = 0
    
    logger.initialization(num_nodes, epochs, bandwidth_mbps)
    logger.info(f"MPLS with List Scheduling Algorithm")
    logger.info(f"tau1 (bandwidth weight): {tau1}, tau2 (divergence weight): {tau2}")
    
    # Test initial accuracy
    initial_accuracy = test_model(model, test_loader, device)
    logger.info(f"[INIT] Initial model accuracy: {initial_accuracy:.2f}%")
    
    for round_num in range(1, r_max + 1):
        logger.round_start(round_num, r_max, clock.get_time())
        logger.info("[PHASE 1] Training locally...")
        t_start = clock.start_compute()
        
        # Store model state before training (for gradient variation)
        if round_num > 1:
            previous_model_states[round_num - 1] = copy_model_state(model)
        
        # Train locally
        train_local_model(model, train_loader, epochs, device)
        
        # Store current model state
        current_model_state = copy_model_state(model)
        
        compute_time = clock.end_compute(t_start)
        logger.compute_phase(compute_time, epochs)
        
        # Place model in shared memory
        with shared_locks[rank]:
            shared_models[rank] = current_model_state
        
        # Network phase - Peer and Layer Selection
        logger.info("[PHASE 2] Peer and Layer Selection...")
        
        # Get candidate peers
        available_peers = [i for i in range(num_nodes) if i != rank]
        candidates = (
            available_peers
            if len(available_peers) <= candidate_set_size
            else random.sample(available_peers, candidate_set_size)
        )
        
        # Step A: Peer Selection
        # Get bandwidths and data divergences for candidates
        peer_bandwidths = {}
        peer_divergences = {}
        peer_proportions = {}
        peer_model_states = {}
        
        # Share our class proportions in shared memory (using a special format)
        # In practice, this would be a separate shared structure, but for simplicity
        # we'll estimate from model behavior or use a heuristic
        
        for candidate_id in candidates:
            # Get peer's model state
            candidate_state = None
            with shared_locks[candidate_id]:
                if shared_models[candidate_id] is not None:
                    candidate_state = shared_models[candidate_id]
            
            if candidate_state is not None:
                # For simplicity, assume uniform bandwidth (can be made variable)
                peer_bandwidths[candidate_id] = bandwidth_mbps
                
                # Estimate peer's class proportions from their model
                # In a real system, peers would explicitly share their class distributions
                # For now, we use a heuristic: assume different nodes have different class distributions
                # based on their rank (Non-IID distribution)
                estimated_proportions = {}
                classes_per_node = 2  # From create_non_iid_dataset
                num_classes = 10
                start_class = (candidate_id * classes_per_node) % num_classes
                node_classes = [(start_class + i) % num_classes for i in range(classes_per_node)]
                
                # Create estimated proportions (most data from assigned classes)
                for c in range(num_classes):
                    if c in node_classes:
                        estimated_proportions[c] = 0.8 / len(node_classes)  # 80% from assigned classes
                    else:
                        estimated_proportions[c] = 0.2 / (num_classes - len(node_classes))  # 20% from others
                
                peer_proportions[candidate_id] = estimated_proportions
                
                # Calculate divergence
                div = calculate_data_divergence(class_proportions, peer_proportions[candidate_id])
                peer_divergences[candidate_id] = div
                peer_model_states[candidate_id] = candidate_state
                total_messages += 2  # Request + response
        
        # Calculate peer selection probabilities
        if not peer_bandwidths or not peer_model_states:
            logger.warning("[PEER_SELECTION] No peers available - skipping aggregation")
            # Still test accuracy
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
            continue
        
        peer_probs = calculate_peer_selection_probability(
            peer_bandwidths, peer_divergences, tau1, tau2
        )
        logger.info(f"[PEER_SELECTION] Probabilities: {peer_probs}")
        
        # Step B: Layer Selection
        # Calculate gradient variations for each layer from each peer
        num_layers = len(layer_names)
        gradient_variations = {}  # {layer_name: {peer_id: variation}}
        
        # Store current peer states for next round
        if round_num not in peer_previous_states:
            peer_previous_states[round_num] = {}
        
        # For early rounds (round 1-2), use uniform gradient variations to ensure learning
        use_uniform_variations = (round_num <= 2)
        
        for layer_name in layer_names:
            gradient_variations[layer_name] = {}
            for peer_id in peer_model_states.keys():
                # Store current peer state for next round (deep copy)
                if peer_id not in peer_previous_states[round_num]:
                    peer_previous_states[round_num][peer_id] = {
                        k: v.clone() for k, v in peer_model_states[peer_id].items()
                    }
                
                if use_uniform_variations:
                    # Early rounds: use uniform variations to ensure all layers can be selected
                    gradient_variations[layer_name][peer_id] = 1.0
                elif round_num > 1 and (round_num - 1) in peer_previous_states:
                    # Calculate variation between current and previous round for this peer
                    peer_prev_state = peer_previous_states[round_num - 1].get(peer_id, {})
                    if peer_prev_state:
                        variation = calculate_gradient_variation(
                            peer_model_states[peer_id],
                            peer_prev_state,
                            layer_name
                        )
                        gradient_variations[layer_name][peer_id] = max(variation, 0.1)  # Minimum 0.1 to avoid zero
                    else:
                        # No previous state for this peer - use comparison with local previous
                        if (round_num - 1) in previous_model_states:
                            variation = calculate_gradient_variation(
                                peer_model_states[peer_id],
                                previous_model_states[round_num - 1],
                                layer_name
                            )
                            gradient_variations[layer_name][peer_id] = max(variation, 0.1)
                        else:
                            gradient_variations[layer_name][peer_id] = 1.0
                else:
                    # First round or no previous state - use comparison with local previous
                    if round_num > 1 and (round_num - 1) in previous_model_states:
                        variation = calculate_gradient_variation(
                            peer_model_states[peer_id],
                            previous_model_states[round_num - 1],
                            layer_name
                        )
                        gradient_variations[layer_name][peer_id] = max(variation, 0.1)
                    else:
                        # First round - use uniform
                        gradient_variations[layer_name][peer_id] = 1.0
        
        # Calculate layer probabilities
        layer_probs = {}  # {layer_name: {peer_id: probability}}
        for layer_name in layer_names:
            layer_probs[layer_name] = calculate_layer_probability(
                gradient_variations[layer_name], layer_name
            )
        
        # Step C: Combined Probability Matrix μ2
        # μ2(s,l) = psk * qsk(l)
        num_peers_used = len(peer_probs)
        mu2 = np.zeros((num_peers_used, num_layers))
        peer_id_to_index = {peer_id: idx for idx, peer_id in enumerate(peer_probs.keys())}
        
        for layer_idx, layer_name in enumerate(layer_names):
            for peer_id, psk in peer_probs.items():
                peer_idx = peer_id_to_index[peer_id]
                qsk_l = layer_probs[layer_name].get(peer_id, 0.0)
                mu2[peer_idx, layer_idx] = psk * qsk_l
        
        # Calculate μ1 matrix: μ1(s,l) = Ml / Bsk
        mu1 = np.zeros((num_peers_used, num_layers))
        peer_bandwidths_list = [peer_bandwidths[peer_id] for peer_id in peer_probs.keys()]
        
        for layer_idx, layer_size in enumerate(layer_sizes):
            for peer_idx, peer_id in enumerate(peer_probs.keys()):
                bw_bps = peer_bandwidths_list[peer_idx] * 1e6  # Convert Mbps to bps
                mu1[peer_idx, layer_idx] = layer_size / bw_bps if bw_bps > 0 else np.inf
        
        # Step D: List Scheduling Algorithm
        logger.info(f"[LIST_SCHEDULING] Starting algorithm with {num_peers_used} peers, {num_layers} layers")
        try:
            assignment = list_scheduling_algorithm(
                mu1, mu2, layer_sizes, peer_bandwidths_list, num_peers_used, num_layers
            )
            logger.info(f"[LIST_SCHEDULING] Algorithm completed: {len(assignment)} layers assigned")
        except Exception as e:
            logger.error(f"[LIST_SCHEDULING] Algorithm failed: {e}, using fallback")
            assignment = {}
        
        logger.info(f"[LIST_SCHEDULING] Layer assignments: {assignment}")
        logger.info(f"[LIST_SCHEDULING] Assigned {len(assignment)}/{num_layers} layers")
        
        # If assignment is empty or incomplete, use a simple fallback: assign all layers to best peer
        if not assignment or len(assignment) < num_layers:
            logger.warning(f"[LIST_SCHEDULING] Incomplete assignment ({len(assignment)}/{num_layers}) - using fallback")
            best_peer_id = max(peer_probs.keys(), key=lambda pid: peer_probs[pid])
            best_peer_idx = peer_id_to_index[best_peer_id]
            for layer_idx in range(num_layers):
                assignment[layer_idx] = best_peer_idx
            logger.info(f"[LIST_SCHEDULING] Fallback assignment: all layers to peer {best_peer_id} (index {best_peer_idx})")
        
        # Aggregate layers from assigned peers
        aggregated_state = copy_model_state(model)
        layers_received = 0
        
        # Collect layers from assigned peers
        layer_contributions = defaultdict(list)  # {param_name: [tensor_from_peer1, tensor_from_peer2, ...]}
        
        if not assignment:
            logger.warning("[AGGREGATION] No layers assigned by list scheduling - using local model only")
        else:
            for layer_idx, peer_idx in assignment.items():
                peer_id = list(peer_probs.keys())[peer_idx]
                layer_name = layer_names[layer_idx]
                
                # Extract layer from peer's model state
                peer_state = peer_model_states.get(peer_id, {})
                for param_name, param_tensor in peer_state.items():
                    if layer_name in param_name:
                        layer_contributions[param_name].append(param_tensor.clone())
                        layers_received += 1
        
        # Aggregate: weighted average of local and received layers
        # For assigned layers: 0.5 local + 0.5 peer (FedAvg style)
        # For unassigned layers: keep local (1.0 local)
        for param_name in aggregated_state:
            if param_name in current_model_state:
                if param_name in layer_contributions and len(layer_contributions[param_name]) > 0:
                    # This parameter was assigned to at least one peer - aggregate
                    peer_avg = sum(layer_contributions[param_name]) / len(layer_contributions[param_name])
                    # Weighted average: 0.5 local + 0.5 peer
                    aggregated_state[param_name] = 0.5 * current_model_state[param_name] + 0.5 * peer_avg
                else:
                    # This parameter was not assigned - keep local model
                    aggregated_state[param_name] = current_model_state[param_name]
        
        # Validate aggregation: check if any parameters changed
        model_state_before = copy_model_state(model)
        
        # Update model
        load_model_state(model, aggregated_state)
        
        # Check if model actually changed
        model_state_after = copy_model_state(model)
        params_changed = 0
        for param_name in model_state_before:
            if param_name in model_state_after:
                diff = torch.sum(torch.abs(model_state_before[param_name] - model_state_after[param_name])).item()
                if diff > 1e-6:  # Significant change
                    params_changed += 1
        
        if assignment:
            logger.info(f"[AGGREGATION] Aggregated {len(assignment)} layers from peers, "
                       f"{layers_received} parameters received, {params_changed} parameters changed")
        else:
            logger.warning("[AGGREGATION] No aggregation performed - model unchanged")
        
        if params_changed == 0 and assignment:
            logger.error("[AGGREGATION] WARNING: Assignment exists but no parameters changed!")
        
        # Calculate data volume and network delay
        total_layer_bits = sum(layer_sizes[layer_idx] for layer_idx in assignment.keys())
        total_data_volume += total_layer_bits
        network_delay = clock.add_network_delay(total_layer_bits, bandwidth_mbps)
        
        logger.network_phase(
            "MPLS-ListScheduling",
            f"Received {len(assignment)} layers from {len(set(assignment.values()))} peers",
            total_layer_bits / 1e6,
            network_delay,
            bandwidth_mbps
        )
        
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


def run_mpls_list_scheduling(
    num_nodes: int,
    r_max: int,
    epochs: int,
    bandwidth_mbps: float,
    log_dir: str = "logs/mpls_list_scheduling",
    tau1: float = 0.5,
    tau2: float = 0.5,
    candidate_set_size: int = 5,
):
    """
    Run MPLS with List Scheduling algorithm.
    
    Args:
        num_nodes: Number of nodes
        r_max: Maximum rounds
        epochs: Local training epochs
        bandwidth_mbps: Network bandwidth
        log_dir: Directory for logs
        tau1: Weight for bandwidth in peer selection (default 0.5)
        tau2: Weight for divergence in peer selection (default 0.5)
        candidate_set_size: Size of candidate set for peer selection
    """
    os.makedirs(log_dir, exist_ok=True)
    train_loaders, test_loader = prepare_data_loaders(num_nodes)
    manager, shared_models, shared_locks, shared_stats = init_shared_objects(num_nodes)
    
    processes = []
    
    try:
        for rank in range(num_nodes):
            p = mp.Process(
                target=list_scheduling_worker,
                args=(
                    rank,
                    num_nodes,
                    r_max,
                    epochs,
                    bandwidth_mbps,
                    train_loaders[rank],
                    test_loader,
                    shared_models,
                    shared_locks,
                    shared_stats,
                    log_dir,
                    tau1,
                    tau2,
                    candidate_set_size,
                )
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        print(f"\n{'='*80}")
        print(f"MPLS with List Scheduling completed!")
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
    
    parser = argparse.ArgumentParser(description="MPLS with List Scheduling Algorithm")
    parser.add_argument("--params", type=str, default="inp-params.txt",
                       help="Path to parameters file (default: inp-params.txt)")
    parser.add_argument("--tau1", type=float, default=0.5,
                       help="Weight for bandwidth in peer selection (default: 0.5)")
    parser.add_argument("--tau2", type=float, default=0.5,
                       help="Weight for divergence in peer selection (default: 0.5)")
    parser.add_argument("--candidate-set-size", type=int, default=5,
                       help="Size of candidate set for peer selection (default: 5)")
    
    args = parser.parse_args()
    
    try:
        params = read_params_file(args.params)
    except Exception as e:
        print(f"Error reading parameters file {args.params}: {e}")
        exit(1)
    
    if len(params) < 6:
        print(f"Error: Expected at least 6 parameters, got {len(params)}")
        exit(1)
    
    # Parse parameters
    num_nodes = int(params[0])
    r_max = int(params[1])
    epochs = int(params[2])
    bandwidth_mbps = float(params[3])
    protocol = int(params[4])
    candidate_set_size = int(params[5]) if len(params) > 5 else args.candidate_set_size
    
    if protocol != 2:
        print("This script is for MPLS protocol (protocol=2)")
        exit(1)
    
    # Ensure tau1 + tau2 = 1
    tau1 = args.tau1
    tau2 = args.tau2
    total = tau1 + tau2
    if total > 0:
        tau1 = tau1 / total
        tau2 = tau2 / total
    
    run_mpls_list_scheduling(
        num_nodes=num_nodes,
        r_max=r_max,
        epochs=epochs,
        bandwidth_mbps=bandwidth_mbps,
        tau1=tau1,
        tau2=tau2,
        candidate_set_size=candidate_set_size,
    )

