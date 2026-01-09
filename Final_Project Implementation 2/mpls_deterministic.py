"""
MPLS Variant 2: Deterministic pairing with global coordination.
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


def await_pair_assignment(
    rank,
    round_num,
    num_nodes,
    ready_queue,
    shared_pairings,
    pairing_lock,
):
    """
    Deterministically pair nodes once everyone is ready for the current round.
    """
    assigned_peer = None
    while assigned_peer is None:
        with pairing_lock:
            queued = list(ready_queue)
            if rank not in queued:
                ready_queue.append(rank)
                queued.append(rank)

            if len(queued) == num_nodes:
                shuffled = list(queued)
                random.shuffle(shuffled)
                shared_pairings.clear()
                for i in range(0, num_nodes, 2):
                    a = shuffled[i]
                    b = shuffled[i + 1]
                    shared_pairings[(round_num, a)] = b
                    shared_pairings[(round_num, b)] = a
                # clear queue
                while len(ready_queue) > 0:
                    ready_queue.pop()

            pair_key = (round_num, rank)
            if pair_key in shared_pairings:
                assigned_peer = shared_pairings.pop(pair_key)

        if assigned_peer is None:
            time.sleep(0.05)
    return assigned_peer


def deterministic_worker(
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
    ready_queue,
    pairing_map,
    pairing_lock,
    log_dir,
):
    device = "cpu"
    clock = HybridClock()
    random.seed(rank + 99)
    torch.manual_seed(rank + 99)
    logger = create_process_logger(rank, log_dir, "MPLS-Deterministic")
    model = LeNet5(num_classes=10).to(device)
    partial_layers = ["fc2", "fc3"]
    stats = {"rounds": [], "virtual_times": [], "accuracies": [], "data_volume_mb": [], "messages": []}
    total_data_volume = 0.0
    total_messages = 0
    logger.initialization(num_nodes, epochs, bandwidth_mbps)
    logger.info(f"Deterministic pairing with {num_nodes} nodes")

    for round_num in range(1, r_max + 1):
        logger.round_start(round_num, r_max, clock.get_time())
        t_start = clock.start_compute()
        train_local_model(model, train_loader, epochs, device)
        local_state = copy_model_state(model)
        with shared_locks[rank]:
            shared_models[rank] = local_state
        partial_state = copy_model_state(model, partial_layers)
        partial_bits = get_model_size_bits(model, partial_layers)
        compute_time = clock.end_compute(t_start)
        logger.compute_phase(compute_time, epochs)

        peer = await_pair_assignment(rank, round_num, num_nodes, ready_queue, pairing_map, pairing_lock)
        logger.peer_selection("Deterministic Pairing", peer)

        metadata_bits = 1024 * candidate_set_size * 2
        payload_bits = partial_bits + metadata_bits
        network_delay = clock.add_network_delay(payload_bits, bandwidth_mbps)
        total_data_volume += payload_bits
        total_messages += 2 + (2 * candidate_set_size)
        logger.network_phase(
            "MPLS-Deterministic", "Sending partial model + metadata", payload_bits / 1e6, network_delay, bandwidth_mbps, peer
        )

        with shared_locks[rank]:
            shared_models[rank] = partial_state

        peer_state = None
        wait_start = time.time()
        while peer_state is None and (time.time() - wait_start) < 30.0:
            with shared_locks[peer]:
                candidate = shared_models[peer]
                if candidate is not None and isinstance(candidate, dict):
                    peer_state = candidate
                    shared_models[peer] = None
            if peer_state is None:
                time.sleep(0.1)

        if peer_state is None:
            logger.error(f"Round {round_num}: peer p{peer} did not deliver partial model - skipping")
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
        logger.info("Deterministic aggregation completed")

        network_delay = clock.add_network_delay(partial_bits, bandwidth_mbps)
        total_data_volume += partial_bits
        logger.network_phase(
            "MPLS-Deterministic", "Receiving partial model", partial_bits / 1e6, network_delay, bandwidth_mbps, peer
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
    logger.info(
        f"Process {rank} finished deterministic pairing run; Accuracy={stats['accuracies'][-1]:.2f}%"
        if stats["accuracies"]
        else f"Process {rank} finished with no accuracy measurements"
    )
    logger.info("=" * 80)


def run_mpls_deterministic(
    num_nodes: int,
    r_max: int,
    epochs: int,
    bandwidth_mbps: float,
    candidate_set_size: int,
    log_dir: str = "logs/mpls_deterministic",
):
    os.makedirs(log_dir, exist_ok=True)
    train_loaders, test_loader = prepare_data_loaders(num_nodes)
    manager, shared_models, shared_locks, shared_stats = init_shared_objects(num_nodes)
    ready_queue = manager.list()
    pairing_map = manager.dict()
    pairing_lock = manager.Lock()

    processes = []
    for rank in range(num_nodes):
        p = mp.Process(
            target=deterministic_worker,
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
                ready_queue,
                pairing_map,
                pairing_lock,
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

    parser = argparse.ArgumentParser(description="Run MPLS deterministic pairing variant")
    parser.add_argument("--params", default="inp-mpls.txt", help="Input parameter file")
    args = parser.parse_args()
    params = read_params_file(args.params)
    N = int(params[0])
    R_max = int(params[1])
    E = int(params[2])
    B = float(params[3])
    C = int(params[5]) if len(params) > 5 else 5
    stats = run_mpls_deterministic(N, R_max, E, B, C)
    print("MPLS Deterministic completed!")
    if stats["accuracies"]:
        print(f"Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        print(f"Total Virtual Time: {stats['virtual_times'][-1]:.2f}s")

