"""
Utility functions for Federated Learning simulation
Includes data loading, model definition, and hybrid clock mechanism
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time
import pickle
import os
import ssl
from datetime import datetime
from typing import Tuple, Dict, List, Optional

# Handle SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context


class ProcessLogger:
    """Enhanced logger for each process with timestamps and file logging"""
    def __init__(self, rank: int, log_dir: str, protocol_name: str):
        self.rank = rank
        self.protocol_name = protocol_name
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'p{rank}_log.txt')
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize log file with header"""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Process {self.rank} - {self.protocol_name} Protocol\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def _log(self, level: str, message: str):
        """Internal logging method with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] [p{self.rank}] [{level}] {message}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        # Also print to console for monitoring
        print(f"[p{self.rank}] [{level}] {message}")
    
    def info(self, message: str):
        """Log info message"""
        self._log("INFO", message)
    
    def debug(self, message: str):
        """Log debug message"""
        self._log("DEBUG", message)
    
    def warning(self, message: str):
        """Log warning message"""
        self._log("WARN", message)
    
    def error(self, message: str):
        """Log error message"""
        self._log("ERROR", message)
    
    def round_start(self, round_num: int, total_rounds: int, virtual_time: float):
        """Log round start"""
        self.info(f"=== Round {round_num}/{total_rounds} STARTED ===")
        self.info(f"Current Virtual Time: {virtual_time:.4f}s")
    
    def round_end(self, round_num: int, virtual_time: float, accuracy: float):
        """Log round end"""
        self.info(f"=== Round {round_num} COMPLETED ===")
        self.info(f"New Virtual Time: {virtual_time:.4f}s | Accuracy: {accuracy:.2f}%")
        self.info("-" * 80)
    
    def compute_phase(self, compute_time: float, epochs: int):
        """Log compute phase"""
        self.info(f"[COMPUTE] Local training completed ({epochs} epochs)")
        self.info(f"[COMPUTE] Real CPU Time: {compute_time:.4f}s")
    
    def network_phase(self, protocol: str, action: str, payload_mbits: float, 
                     delay: float, bandwidth: float, peer: Optional[int] = None):
        """Log network phase"""
        peer_str = f" with p{peer}" if peer is not None else ""
        self.info(f"[NETWORK] Protocol: {protocol} | Action: {action}{peer_str}")
        self.info(f"[NETWORK] Payload: {payload_mbits:.4f} Mbits | Delay: {delay:.4f}s | Bandwidth: {bandwidth} Mbps")
    
    def aggregation(self, num_models: int, method: str = "FedAvg"):
        """Log aggregation"""
        self.info(f"[AGGREGATION] Aggregating {num_models} models using {method}")
    
    def peer_selection(self, method: str, selected_peer: int, candidates: Optional[list] = None):
        """Log peer selection"""
        if candidates:
            self.info(f"[PEER_SELECTION] Method: {method} | Candidates: {candidates} | Selected: p{selected_peer}")
        else:
            self.info(f"[PEER_SELECTION] Method: {method} | Selected: p{selected_peer}")
    
    def statistics(self, round_num: int, virtual_time: float, accuracy: float, 
                  data_volume_mb: float, messages: int):
        """Log statistics"""
        self.info(f"[STATS] Round: {round_num} | Virtual Time: {virtual_time:.4f}s | "
                 f"Accuracy: {accuracy:.2f}% | Data: {data_volume_mb:.4f} MB | Messages: {messages}")
    
    def initialization(self, num_nodes: int, epochs: int, bandwidth: float):
        """Log initialization"""
        self.info(f"[INIT] Process {self.rank} initialized")
        self.info(f"[INIT] System: {num_nodes} nodes | Local epochs: {epochs} | Bandwidth: {bandwidth} Mbps")
        self.info(f"[INIT] Model: LeNet-5 | Dataset: MNIST (Non-IID)")
    
    def synchronization(self, phase: str, status: str):
        """Log synchronization events"""
        self.debug(f"[SYNC] Phase: {phase} | Status: {status}")


class LeNet5(nn.Module):
    """LeNet-5 model architecture"""
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_layer_names(self):
        """Return list of layer names for partial model transfer"""
        return ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']


class HybridClock:
    """Hybrid Clock mechanism: Real Compute Time + Simulated Network Time"""
    def __init__(self):
        self.virtual_time = 0.0
        self.compute_time = 0.0
        self.network_time = 0.0
        
    def start_compute(self):
        """Start measuring real CPU time"""
        return time.process_time()
    
    def end_compute(self, start_time):
        """End compute measurement and update virtual time"""
        elapsed = time.process_time() - start_time
        self.compute_time += elapsed
        self.virtual_time += elapsed
        return elapsed
    
    def add_network_delay(self, payload_bits, bandwidth_mbps):
        """Add simulated network delay based on payload size and bandwidth"""
        # Convert Mbps to bits per second
        bandwidth_bps = bandwidth_mbps * 1e6
        delay = payload_bits / bandwidth_bps
        self.network_time += delay
        self.virtual_time += delay
        return delay
    
    def get_time(self):
        """Get current virtual time"""
        return self.virtual_time
    
    def reset(self):
        """Reset clock"""
        self.virtual_time = 0.0
        self.compute_time = 0.0
        self.network_time = 0.0


def create_non_iid_dataset(num_nodes: int, num_classes: int = 10, 
                           classes_per_node: int = 2, seed: int = 42) -> List[Subset]:
    """
    Create Non-IID data distribution (label skew)
    Each node gets only a subset of classes
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                  transform=transform)
    
    # Group indices by class
    indices_by_class = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(train_dataset):
        # Handle both tensor and int labels
        label_val = label.item() if hasattr(label, 'item') else label
        indices_by_class[label_val].append(idx)
    
    # Shuffle each class's indices
    for class_idx in indices_by_class:
        np.random.shuffle(indices_by_class[class_idx])
    
    # Assign classes to nodes (Non-IID)
    node_datasets = []
    class_assignments = []
    
    # Distribute classes to nodes
    for node_id in range(num_nodes):
        # Assign classes_per_node classes to this node
        start_class = (node_id * classes_per_node) % num_classes
        node_classes = [(start_class + i) % num_classes for i in range(classes_per_node)]
        class_assignments.append(node_classes)
        
        # Collect indices for these classes
        node_indices = []
        for class_idx in node_classes:
            # Distribute samples of this class across nodes that have it
            class_size = len(indices_by_class[class_idx])
            nodes_with_class = num_nodes // (num_classes // classes_per_node)
            per_node_size = class_size // nodes_with_class
            start_idx = (node_id % nodes_with_class) * per_node_size
            end_idx = start_idx + per_node_size
            node_indices.extend(indices_by_class[class_idx][start_idx:end_idx])
        
        node_datasets.append(Subset(train_dataset, node_indices))
    
    return node_datasets, test_dataset, class_assignments


def get_model_size_bits(model: nn.Module, partial_layers: List[str] = None) -> int:
    """
    Calculate model size in bits
    If partial_layers is provided, only count those layers
    """
    total_bits = 0
    state_dict = model.state_dict()
    
    if partial_layers is None:
        # Full model
        for param in state_dict.values():
            total_bits += param.numel() * 32  # 32 bits per float32
    else:
        # Partial model
        for layer_name in partial_layers:
            for key in state_dict.keys():
                if layer_name in key:
                    total_bits += state_dict[key].numel() * 32
    
    return total_bits


def get_payload_size_bits(model: nn.Module, partial_layers: List[str] = None, 
                         metadata_size: int = 0) -> int:
    """
    Calculate total payload size in bits including metadata
    """
    model_bits = get_model_size_bits(model, partial_layers)
    return model_bits + metadata_size


def train_local_model(model: nn.Module, train_loader: DataLoader, 
                     epochs: int, device: str = 'cpu') -> float:
    """
    Train model locally for specified epochs
    Returns average loss
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_local_model_prox(model: nn.Module, train_loader: DataLoader, epochs: int,
                           global_state: Dict[str, torch.Tensor], mu: float,
                           device: str = 'cpu') -> float:
    """
    Train model locally with FedProx proximal term
    global_state: reference parameters (typically aggregated global model)
    mu: proximal coefficient
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Move global reference tensors to training device once
    ref_state = {k: v.to(device) for k, v in global_state.items()}
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Add proximal term (mu/2) * ||w - w_ref||^2
            prox_term = 0.0
            for name, param in model.named_parameters():
                if name in ref_state:
                    prox_term = prox_term + torch.sum((param - ref_state[name]) ** 2)
            loss = loss + (mu / 2.0) * prox_term
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def test_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> float:
    """
    Test model and return accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100.0 * correct / total if total > 0 else 0.0


def aggregate_models(models: List[nn.Module], weights: List[float] = None) -> nn.Module:
    """
    Aggregate multiple models using weighted average
    If weights is None, use uniform weights
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Get first model as base
    aggregated = models[0]
    aggregated_state = aggregated.state_dict()
    
    # Initialize aggregated state
    for key in aggregated_state:
        aggregated_state[key] = aggregated_state[key] * weights[0]
    
    # Aggregate remaining models
    for i, model in enumerate(models[1:], 1):
        state = model.state_dict()
        for key in aggregated_state:
            aggregated_state[key] += state[key] * weights[i]
    
    # Load aggregated state
    aggregated.load_state_dict(aggregated_state)
    return aggregated


def copy_model_state(model: nn.Module, partial_layers: List[str] = None) -> Dict:
    """
    Copy model state (full or partial)
    """
    state_dict = model.state_dict()
    if partial_layers is None:
        return {k: v.clone() for k, v in state_dict.items()}
    else:
        result = {}
        for layer_name in partial_layers:
            for key in state_dict.keys():
                if layer_name in key:
                    result[key] = state_dict[key].clone()
        return result


def load_model_state(model: nn.Module, state_dict: Dict, partial_layers: List[str] = None):
    """
    Load state into model (full or partial)
    """
    model_state = model.state_dict()
    if partial_layers is None:
        model_state.update(state_dict)
    else:
        for key in state_dict:
            if key in model_state:
                model_state[key] = state_dict[key]
    model.load_state_dict(model_state)


def read_params_file(path: str) -> List[str]:
    """
    Read the first non-empty, non-comment line from a params file.
    Returns the comma-separated values as a list of strings.
    """
    with open(path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            return [token.strip() for token in line.split(',')]
    raise ValueError(f"No parameter values found in {path}")

