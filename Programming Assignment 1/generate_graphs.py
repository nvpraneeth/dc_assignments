#!/usr/bin/env python3
"""
Script to generate performance comparison graphs for Vector Clock vs Singhal-Kshemkalyani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def generate_performance_graphs():
    """Generate all required graphs for the report"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    try:
        # Read the CSV data
        df = pd.read_csv('performance_data.csv')
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Vector Clock vs Singhal-Kshemkalyani Performance Comparison', fontsize=16, fontweight='bold')
        
        # Graph 1: Average entries per message vs Number of processes
        ax1.plot(df['Processes'], df['VC_Avg_Entries'], 'o-', label='Vector Clock', linewidth=2, markersize=8)
        ax1.plot(df['Processes'], df['SK_Avg_Entries'], 's-', label='Singhal-Kshemkalyani', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Processes')
        ax1.set_ylabel('Average Entries per Message')
        ax1.set_title('Message Overhead Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add percentage improvement annotations
        for i, row in df.iterrows():
            improvement = ((row['VC_Avg_Entries'] - row['SK_Avg_Entries']) / row['VC_Avg_Entries']) * 100
            ax1.annotate(f'{improvement:.1f}%', 
                        xy=(row['Processes'], row['SK_Avg_Entries']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Graph 2: Total messages sent
        ax2.plot(df['Processes'], df['VC_Messages'], 'o-', label='Vector Clock', linewidth=2, markersize=8)
        ax2.plot(df['Processes'], df['SK_Messages'], 's-', label='Singhal-Kshemkalyani', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Processes')
        ax2.set_ylabel('Total Messages Sent')
        ax2.set_title('Total Message Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Graph 3: Execution time comparison
        ax3.plot(df['Processes'], df['VC_Time'], 'o-', label='Vector Clock', linewidth=2, markersize=8)
        ax3.plot(df['Processes'], df['SK_Time'], 's-', label='Singhal-Kshemkalyani', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Processes')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Execution Time Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Graph 4: Improvement percentage
        improvements = []
        for _, row in df.iterrows():
            improvement = ((row['VC_Avg_Entries'] - row['SK_Avg_Entries']) / row['VC_Avg_Entries']) * 100
            improvements.append(improvement)
        
        ax4.bar(df['Processes'], improvements, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax4.set_xlabel('Number of Processes')
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('Message Size Reduction with SK Optimization')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(improvements):
            ax4.text(df['Processes'].iloc[i], v + 0.5, f'{v:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('performance_comparison_graphs.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate detailed analysis report
        generate_analysis_report(df)
        
    except FileNotFoundError:
        print("Error: performance_data.csv not found. Please run the performance comparison first.")
    except Exception as e:
        print(f"Error generating graphs: {e}")

def generate_analysis_report(df):
    """Generate a detailed analysis report"""
    
    report = []
    report.append("PERFORMANCE ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS:")
    report.append("-" * 20)
    
    avg_vc_entries = df['VC_Avg_Entries'].mean()
    avg_sk_entries = df['SK_Avg_Entries'].mean()
    avg_improvement = ((avg_vc_entries - avg_sk_entries) / avg_vc_entries) * 100
    
    report.append(f"Average Vector Clock entries per message: {avg_vc_entries:.2f}")
    report.append(f"Average SK entries per message: {avg_sk_entries:.2f}")
    report.append(f"Average improvement: {avg_improvement:.2f}%")
    report.append("")
    
    # Detailed analysis for each process count
    report.append("DETAILED ANALYSIS BY PROCESS COUNT:")
    report.append("-" * 40)
    
    for _, row in df.iterrows():
        improvement = ((row['VC_Avg_Entries'] - row['SK_Avg_Entries']) / row['VC_Avg_Entries']) * 100
        time_ratio = row['SK_Time'] / row['VC_Time'] if row['VC_Time'] > 0 else 0
        
        report.append(f"Processes: {row['Processes']}")
        report.append(f"  VC Entries: {row['VC_Avg_Entries']:.2f}")
        report.append(f"  SK Entries: {row['SK_Avg_Entries']:.2f}")
        report.append(f"  Improvement: {improvement:.2f}%")
        report.append(f"  Time Ratio (SK/VC): {time_ratio:.3f}")
        report.append("")
    
    # Anomaly detection
    report.append("ANOMALY ANALYSIS:")
    report.append("-" * 20)
    
    # Check for unexpected patterns
    entries_diff = df['VC_Avg_Entries'] - df['SK_Avg_Entries']
    if (entries_diff < 0).any():
        report.append("WARNING: Some cases show SK sending more entries than VC!")
        report.append("This might indicate an implementation issue.")
    else:
        report.append("✓ SK optimization consistently reduces message size")
    
    # Check if improvement increases with process count
    improvements = []
    for _, row in df.iterrows():
        improvement = ((row['VC_Avg_Entries'] - row['SK_Avg_Entries']) / row['VC_Avg_Entries']) * 100
        improvements.append(improvement)
    
    if len(improvements) > 1:
        improvement_trend = np.polyfit(df['Processes'], improvements, 1)[0]
        if improvement_trend > 0:
            report.append("✓ Improvement increases with process count (expected)")
        else:
            report.append("⚠ Improvement decreases with process count (unexpected)")
    
    # Write report to file
    with open('analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("Analysis report saved to analysis_report.txt")

if __name__ == "__main__":
    generate_performance_graphs()
