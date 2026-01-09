"""
Decentralized Handshake: Smart peer selection with handshake-based exchange.
"""

import os
import random
import time

import torch
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


def calculate_model_similarity(model1: LeNet5, model2: LeNet5, layer_name: str = "fc3") -> float:
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    layer_key = None
    for key in state1.keys():
        if layer_name in key and "weight" in key:
            layer_key = key
            break
    if layer_key is None:
        return 0.0
    w1 = state1[layer_key].flatten()
    w2 = state2[layer_key].flatten()
    dot_product = torch.dot(w1, w2)
    norm1 = torch.norm(w1)
    norm2 = torch.norm(w2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return (dot_product / (norm1 * norm2)).item()


def select_smart_peer(rank, num_nodes, model, candidate_set_size, shared_models, shared_locks):
    available_peers = [i for i in range(num_nodes) if i != rank]
    candidates = (
        available_peers
        if len(available_peers) <= candidate_set_size
        else random.sample(available_peers, candidate_set_size)
    )
    similarities = []
    for candidate_id in candidates:
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
    if similarities:
        return min(similarities, key=lambda x: x[1])[0]
    return random.choice(available_peers)


def handshake_worker(
    rank,
    num_nodes,
    r_max,
    epochs,
    bandwidth_mbps,
    candidate_set_size,
    train_loader: DataLoader,
    test_loader: DataLoader,
    shared_models,
    shared_locks,
    shared_stats,
    log_dir,
):
    device = "cpu"
    clock = HybridClock()
    random.seed(rank + 42)
    torch.manual_seed(rank + 42)
    logger = create_process_logger(rank, log_dir, "Decentralized-Handshake")
    model = LeNet5(num_classes=10).to(device)
    partial_layers = ["fc2", "fc3"]
    stats = {"rounds": [], "virtual_times": [], "accuracies": [], "data_volume_mb": [], "messages": []}
    total_data_volume = 0.0
    total_messages = 0
    logger.initialization(num_nodes, epochs, bandwidth_mbps)
    logger.info(f"Partial layers: {partial_layers}")
    logger.info(f"Candidate set size: {candidate_set_size}")

    for round_num in range(1, r_max + 1):
        logger.round_start(round_num, r_max, clock.get_time())
        logger.info("[PHASE 1] Training locally...")
        t_start = clock.start_compute()
        train_local_model(model, train_loader, epochs, device)
        local_state = copy_model_state(model)
        with shared_locks[rank]:
            shared_models[rank] = local_state
        selected_peer = select_smart_peer(rank, num_nodes, model, candidate_set_size, shared_models, shared_locks)
        partial_state = copy_model_state(model, partial_layers)
        partial_bits = get_model_size_bits(model, partial_layers)
        compute_time = clock.end_compute(t_start)
        logger.compute_phase(compute_time, epochs)

        metadata_bits = 1024 * candidate_set_size * 2
        payload_bits = partial_bits + metadata_bits
        network_delay = clock.add_network_delay(payload_bits, bandwidth_mbps)
        total_data_volume += payload_bits
        total_messages += 2 + (2 * candidate_set_size)
        logger.network_phase(
            "Decentralized-Handshake", "Sending partial model + metadata", payload_bits / 1e6, network_delay, bandwidth_mbps, selected_peer
        )

        with shared_locks[rank]:
            shared_models[rank] = partial_state
        logger.synchronization("Upload", f"Placed partial model for p{selected_peer}")

        peer_state = None
        wait_start = time.time()
        while peer_state is None and (time.time() - wait_start) < 30.0:
            with shared_locks[selected_peer]:
                candidate = shared_models[selected_peer]
                if candidate is not None and isinstance(candidate, dict):
                    peer_state = candidate
                    shared_models[selected_peer] = None
            if peer_state is None:
                time.sleep(0.2)
        if peer_state is None:
            logger.error(f"Timeout waiting for partial model from p{selected_peer} - skipping round")
            with shared_locks[rank]:
                shared_models[rank] = None
            stats["rounds"].append(round_num)
            stats["virtual_times"].append(clock.get_time())
            stats["accuracies"].append(0.0)
            stats["data_volume_mb"].append(total_data_volume / (8 * 1e6))
            stats["messages"].append(total_messages)
            continue

        with shared_locks[rank]:
            shared_models[rank] = None

        peer_model = LeNet5(num_classes=10).to(device)
        load_model_state(peer_model, copy_model_state(model))
        load_model_state(peer_model, peer_state, partial_layers)
        current_state = model.state_dict()
        peer_state_dict = peer_model.state_dict()
        for layer_name in partial_layers:
            for key in current_state.keys():
                if layer_name in key:
                    current_state[key] = 0.5 * current_state[key] + 0.5 * peer_state_dict[key]
        model.load_state_dict(current_state)
        logger.info("Partial aggregation completed")

        network_delay = clock.add_network_delay(partial_bits, bandwidth_mbps)
        total_data_volume += partial_bits
        logger.network_phase(
            "Decentralized-Handshake", "Receiving partial model", partial_bits / 1e6, network_delay, bandwidth_mbps, selected_peer
        )

        accuracy = test_model(model, test_loader, device)
        stats["rounds"].append(round_num)
        stats["virtual_times"].append(clock.get_time())
        stats["accuracies"].append(accuracy)
        stats["data_volume_mb"].append(total_data_volume / (8 * 1e6))
        stats["messages"].append(total_messages)
        logger.statistics(round_num, clock.get_time(), accuracy, total_data_volume / (8 * 1e6), total_messages)
        logger.round_end(round_num, clock.get_time(), accuracy)
        if rank == 0:
            shared_stats["rounds"] = stats["rounds"]
            shared_stats["virtual_times"] = stats["virtual_times"]
            shared_stats["accuracies"] = stats["accuracies"]
            shared_stats["data_volume_mb"] = stats["data_volume_mb"]
            shared_stats["messages"] = stats["messages"]

    logger.info("=" * 80)
    logger.info(f"Process {rank} finished all rounds; Accuracy={stats['accuracies'][-1]:.2f}%")
    logger.info("=" * 80)


def run_mpls_handshake(num_nodes: int, r_max: int, epochs: int, bandwidth_mbps: float, candidate_set_size: int, log_dir: str = "logs/mpls_handshake"):
    """
    Run decentralized handshake protocol.
    Note: Function name kept as run_mpls_handshake for backward compatibility.
    """
    os.makedirs(log_dir, exist_ok=True)
    train_loaders, test_loader = prepare_data_loaders(num_nodes)
    manager, shared_models, shared_locks, shared_stats = init_shared_objects(num_nodes)
    processes = []
    for rank in range(num_nodes):
        p = mp.Process(
            target=handshake_worker,
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
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return dict(shared_stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Decentralized Handshake protocol")
    parser.add_argument("--params", default="inp-mpls.txt", help="Input parameter file")
    args = parser.parse_args()
    params = read_params_file(args.params)
    N = int(params[0])
    R_max = int(params[1])
    E = int(params[2])
    B = float(params[3])
    C = int(params[5]) if len(params) > 5 else 5
    stats = run_mpls_handshake(N, R_max, E, B, C)
    print("Decentralized Handshake completed!")
    if stats["accuracies"]:
        print(f"Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        print(f"Total Virtual Time: {stats['virtual_times'][-1]:.2f}s")

