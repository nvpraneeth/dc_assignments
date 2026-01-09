"""
Experiment Runner and Visualization
Runs all three protocols multiple times and generates comparative graphs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from fedavg import run_fedavg
from fedprox import run_fedprox
from decentralized_baseline import run_decentralized_baseline
from decentralized_handshake import run_mpls_handshake
from mpls_deterministic import run_mpls_deterministic
from utils import read_params_file


def run_experiments(num_runs: int = 5):
    """
    Run all three protocols num_runs times and average results
    """
    # Read input parameters
    params = read_params_file('inp-params.txt')
    N = int(params[0])
    R_max = int(params[1])
    E = int(params[2])
    B = float(params[3])
    Protocol = int(params[4])
    C = int(params[5]) if len(params) > 5 else 5
    MU = float(params[6]) if len(params) > 6 else 0.1
    
    print(f"Running experiments with N={N}, R_max={R_max}, E={E}, B={B} Mbps")
    print(f"Number of runs per protocol: {num_runs}\n")
    
    # Storage for all runs
    fedavg_results = []
    fedprox_results = []
    decentralized_results = []
    mpls_handshake_results = []
    mpls_deterministic_results = []
    
    # Run FedAvg
    print("=" * 60)
    print("Running FedAvg (Protocol 0)")
    print("=" * 60)
    for run in range(num_runs):
        print(f"\nFedAvg Run {run + 1}/{num_runs}")
        try:
            stats = run_fedavg(N, R_max, E, B, log_dir=f'logs/fedavg_run{run+1}')
            fedavg_results.append(stats)
            print(f"  Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        except Exception as e:
            print(f"  Error in run {run + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run FedProx
    print("\n" + "=" * 60)
    print("Running FedProx (Protocol 5)")
    print("=" * 60)
    for run in range(num_runs):
        print(f"\nFedProx Run {run + 1}/{num_runs}")
        try:
            stats = run_fedprox(N, R_max, E, B, MU, log_dir=f'logs/fedprox_run{run+1}')
            fedprox_results.append(stats)
            print(f"  Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        except Exception as e:
            print(f"  Error in run {run + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run Decentralized Baseline
    print("\n" + "=" * 60)
    print("Running Decentralized Baseline (Protocol 1)")
    print("=" * 60)
    for run in range(num_runs):
        print(f"\nDecentralized Baseline Run {run + 1}/{num_runs}")
        try:
            stats = run_decentralized_baseline(N, R_max, E, B, log_dir=f'logs/decentralized_run{run+1}')
            decentralized_results.append(stats)
            print(f"  Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        except Exception as e:
            print(f"  Error in run {run + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run MPLS Handshake
    print("\n" + "=" * 60)
    print("Running MPLS Handshake Variant")
    print("=" * 60)
    for run in range(num_runs):
        print(f"\nMPLS Handshake Run {run + 1}/{num_runs}")
        try:
            stats = run_mpls_handshake(N, R_max, E, B, C, log_dir=f'logs/mpls_handshake_run{run+1}')
            mpls_handshake_results.append(stats)
            print(f"  Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        except Exception as e:
            print(f"  Error in run {run + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run MPLS Deterministic Pairing
    print("\n" + "=" * 60)
    print("Running MPLS Deterministic Pairing Variant")
    print("=" * 60)
    for run in range(num_runs):
        print(f"\nMPLS Deterministic Run {run + 1}/{num_runs}")
        try:
            stats = run_mpls_deterministic(N, R_max, E, B, C, log_dir=f'logs/mpls_deterministic_run{run+1}')
            mpls_deterministic_results.append(stats)
            print(f"  Final Accuracy: {stats['accuracies'][-1]:.2f}%")
        except Exception as e:
            print(f"  Error in run {run + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Average results
    print("\n" + "=" * 60)
    print("Averaging results...")
    print("=" * 60)
    
    fedavg_avg = average_results(fedavg_results)
    fedprox_avg = average_results(fedprox_results)
    decentralized_avg = average_results(decentralized_results)
    mpls_handshake_avg = average_results(mpls_handshake_results)
    mpls_deterministic_avg = average_results(mpls_deterministic_results)
    
    series = [
        ("FedAvg (Centralized)", fedavg_avg),
        ("FedProx (Centralized)", fedprox_avg),
        ("Decentralized Baseline", decentralized_avg),
        ("MPLS Handshake", mpls_handshake_avg),
        ("MPLS Deterministic", mpls_deterministic_avg),
    ]
    
    # Generate graphs
    print("\nGenerating comparative graphs...")
    generate_graphs(series)
    
    print("\nExperiments completed! Check 'experiments_output/' directory for graphs.")


def average_results(results_list):
    """
    Average multiple result dictionaries
    """
    if not results_list:
        return None
    
    # Find maximum length
    max_len = max(len(r['rounds']) for r in results_list)
    
    # Initialize averaged results
    avg = {
        'rounds': list(range(1, max_len + 1)),
        'virtual_times': [],
        'accuracies': [],
        'data_volume_mb': [],
        'messages': []
    }
    
    # Average each metric
    for idx in range(max_len):
        accuracies = []
        virtual_times = []
        data_volumes = []
        messages = []
        
        for result in results_list:
            if idx < len(result['accuracies']):
                accuracies.append(result['accuracies'][idx])
                virtual_times.append(result['virtual_times'][idx])
                data_volumes.append(result['data_volume_mb'][idx])
                messages.append(result['messages'][idx])
        
        avg['accuracies'].append(np.mean(accuracies) if accuracies else 0)
        avg['virtual_times'].append(np.mean(virtual_times) if virtual_times else 0)
        avg['data_volume_mb'].append(np.mean(data_volumes) if data_volumes else 0)
        avg['messages'].append(np.mean(messages) if messages else 0)
    
    return avg


def generate_graphs(result_series):
    """
    Generate the four comparative graphs
    """
    os.makedirs('experiments_output', exist_ok=True)
    
    # Experiment 1: Accuracy vs Rounds
    plt.figure(figsize=(10, 6))
    for label, stats in result_series:
        if stats:
            plt.plot(
                stats['rounds'],
                stats['accuracies'],
                label=label,
                marker='o',
                markersize=4,
                linewidth=2,
            )
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Global Test Accuracy (%)', fontsize=12)
    plt.title('Experiment 1: Convergence Speed (Accuracy vs. Rounds)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments_output/exp1_accuracy_vs_rounds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Experiment 1: Accuracy vs Rounds")
    
    # Experiment 2: Accuracy vs Virtual Time
    plt.figure(figsize=(10, 6))
    for label, stats in result_series:
        if stats:
            plt.plot(
                stats['virtual_times'],
                stats['accuracies'],
                label=label,
                marker='o',
                markersize=4,
                linewidth=2,
            )
    plt.xlabel('Virtual Wall Clock Time (Seconds)', fontsize=12)
    plt.ylabel('Global Test Accuracy (%)', fontsize=12)
    plt.title('Experiment 2: Convergence Speed (Accuracy vs. Virtual Time)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments_output/exp2_accuracy_vs_virtual_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Experiment 2: Accuracy vs Virtual Time")
    
    # Experiment 3: Accuracy vs Data Volume
    plt.figure(figsize=(10, 6))
    for label, stats in result_series:
        if stats:
            plt.plot(
                stats['data_volume_mb'],
                stats['accuracies'],
                label=label,
                marker='o',
                markersize=4,
                linewidth=2,
            )
    plt.xlabel('Total MegaBytes Transferred (Upload + Download)', fontsize=12)
    plt.ylabel('Global Test Accuracy (%)', fontsize=12)
    plt.title('Experiment 3: Bandwidth Efficiency (Accuracy vs. Data Volume)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments_output/exp3_accuracy_vs_data_volume.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Experiment 3: Accuracy vs Data Volume")
    
    # Experiment 4: Messages vs Rounds
    plt.figure(figsize=(10, 6))
    for label, stats in result_series:
        if stats:
            plt.plot(
                stats['rounds'],
                stats['messages'],
                label=label,
                marker='o',
                markersize=4,
                linewidth=2,
            )
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Cumulative Number of Messages Exchanged', fontsize=12)
    plt.title('Experiment 4: Message Complexity (Messages vs. Rounds)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments_output/exp4_messages_vs_rounds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Experiment 4: Messages vs Rounds")
    
    # Save numerical results
    save_results(result_series)


def save_results(result_series):
    """
    Save numerical results to files
    """
    with open('experiments_output/results_summary.txt', 'w') as f:
        f.write("Federated Learning Experiments - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        
        for label, stats in result_series:
            if not stats:
                continue
            f.write(f"{label}:\n")
            f.write(f"  Final Accuracy: {stats['accuracies'][-1]:.2f}%\n")
            f.write(f"  Total Virtual Time: {stats['virtual_times'][-1]:.2f}s\n")
            f.write(f"  Total Data Volume: {stats['data_volume_mb'][-1]:.2f} MB\n")
            f.write(f"  Total Messages: {stats['messages'][-1]:.0f}\n\n")


if __name__ == '__main__':
    num_runs = 5
    if len(sys.argv) > 1:
        num_runs = int(sys.argv[1])
    
    run_experiments(num_runs)

