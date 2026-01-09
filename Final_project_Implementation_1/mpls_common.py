"""
Shared helpers for MPLS protocol variants.
"""

import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from utils import (
    create_non_iid_dataset,
    ProcessLogger,
)


def prepare_data_loaders(num_nodes: int):
    """
    Create train/test loaders for MPLS experiments.
    """
    node_datasets, test_dataset, _ = create_non_iid_dataset(num_nodes)
    train_loaders = [
        DataLoader(dataset, batch_size=32, shuffle=True) for dataset in node_datasets
    ]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loaders, test_loader


def init_shared_objects(num_nodes: int, slots_per_node: int = 1):
    """
    Initialize multiprocessing managers, shared model storage and stats.
    """
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    total_slots = num_nodes * slots_per_node
    shared_models = manager.list([None] * total_slots)
    shared_locks = [manager.Lock() for _ in range(total_slots)]
    shared_stats = manager.dict(
        {
            "rounds": [],
            "virtual_times": [],
            "accuracies": [],
            "data_volume_mb": [],
            "messages": [],
        }
    )
    return manager, shared_models, shared_locks, shared_stats


def init_synchronizer_objects(num_nodes: int):
    """
    Initialize shared objects for synchronizer.
    
    Returns:
        (shared_sync, sync_locks) where:
        - shared_sync: Shared dictionary for synchronization state
        - sync_locks: List of locks for synchronization (one per node)
    """
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    
    # Shared synchronization state
    shared_sync = manager.dict({
        'round_num': 0,
        'train_ready': manager.list([False] * num_nodes),
        'model_placed': manager.list([False] * num_nodes),
        'exchange_done': manager.list([False] * num_nodes),
        'round_ready': manager.list([False] * num_nodes),
        'last_update': manager.list([0.0] * num_nodes),
        'health_check': manager.list([0.0] * num_nodes),
    })
    
    # One lock per node for synchronization
    sync_locks = [manager.Lock() for _ in range(num_nodes)]
    
    return shared_sync, sync_locks


def create_process_logger(rank: int, log_dir: str, variant_name: str) -> ProcessLogger:
    """
    Helper to create namespaced loggers for each MPLS variant.
    """
    return ProcessLogger(rank, log_dir, variant_name)

