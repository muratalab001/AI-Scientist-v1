import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_run_data(run_dir):
    """Load data from a specific run directory."""
    try:
        with open(os.path.join(run_dir, "final_info.json"), "r") as f:
            data = json.load(f)
        return data["differential_equation"]
    except (FileNotFoundError, KeyError):
        return None

def plot_solutions():
    """Plot analytical vs numerical solutions for all runs."""
    plt.figure(figsize=(12, 8))
    
    # Define colors for different runs
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Find all run directories
    run_dirs = sorted([d for d in os.listdir('.') if d.startswith('run_') and os.path.isdir(d)])
    
    labels = {}
    for i, run_dir in enumerate(run_dirs):
        data = load_run_data(run_dir)
        if data is None:
            continue
            
        x_vals = np.array(data["x_values"])
        y_analytical = np.array(data["y_analytical"])
        y_numerical = np.array(data["y_numerical"])
        
        color = colors[i % len(colors)]
        
        # Plot analytical solution
        plt.plot(x_vals, y_analytical, '--', color=color, linewidth=2, 
                label=f'{run_dir} (Analytical)')
        
        # Plot numerical solution
        plt.plot(x_vals, y_numerical, '-', color=color, linewidth=1, alpha=0.7,
                label=f'{run_dir} (Numerical)')
        
        # Store label for legend
        labels[run_dir] = f'Run {run_dir.split("_")[1]}'
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Analytical vs Numerical Solutions of Differential Equations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('solutions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_analysis():
    """Plot error analysis for all runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    run_dirs = sorted([d for d in os.listdir('.') if d.startswith('run_') and os.path.isdir(d)])
    
    mse_values = []
    relative_errors = []
    run_labels = []
    
    for run_dir in run_dirs:
        data = load_run_data(run_dir)
        if data is None:
            continue
            
        means = data["means"]
        mse_values.append(means["mse"])
        relative_errors.append(means["relative_error"])
        run_labels.append(f'Run {run_dir.split("_")[1]}')
    
    # Plot MSE
    ax1.bar(run_labels, mse_values, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Mean Squared Error by Run')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot Relative Error
    ax2.bar(run_labels, relative_errors, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Run')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Relative Error by Run')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_solution_characteristics():
    """Plot solution characteristics for all runs."""
    run_dirs = sorted([d for d in os.listdir('.') if d.startswith('run_') and os.path.isdir(d)])
    
    characteristics = {
        'max_value': [],
        'min_value': [],
        'final_value': [],
        'growth_rate': [],
        'convergence_rate': []
    }
    run_labels = []
    
    for run_dir in run_dirs:
        data = load_run_data(run_dir)
        if data is None:
            continue
            
        char = data["means"]["solution_characteristics"]
        for key in characteristics:
            characteristics[key].append(char[key])
        run_labels.append(f'Run {run_dir.split("_")[1]}')
    
    # Create subplots for different characteristics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    char_names = list(characteristics.keys())
    for i, (char_name, values) in enumerate(characteristics.items()):
        if i < len(axes):
            axes[i].bar(run_labels, values, color=plt.cm.viridis(np.linspace(0, 1, len(values))))
            axes[i].set_title(f'{char_name.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Remove empty subplot
    if len(characteristics) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('solution_characteristics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_equation_comparison():
    """Plot different equations side by side for comparison."""
    run_dirs = sorted([d for d in os.listdir('.') if d.startswith('run_') and os.path.isdir(d)])
    
    if len(run_dirs) == 0:
        print("No run data found.")
        return
    
    # Create subplots for each run
    n_runs = len(run_dirs)
    cols = min(3, n_runs)
    rows = (n_runs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_runs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, run_dir in enumerate(run_dirs):
        data = load_run_data(run_dir)
        if data is None:
            continue
            
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        x_vals = np.array(data["x_values"])
        y_analytical = np.array(data["y_analytical"])
        y_numerical = np.array(data["y_numerical"])
        
        ax.plot(x_vals, y_analytical, '--', color='blue', linewidth=2, label='Analytical')
        ax.plot(x_vals, y_numerical, '-', color='red', linewidth=1, alpha=0.7, label='Numerical')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y(x)')
        ax.set_title(f'{run_dir} - {data["original_equation"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_runs, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].remove()
        else:
            axes[col].remove()
    
    plt.tight_layout()
    plt.savefig('equation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating plots for differential equation experiments...")
    
    # Generate all plots
    plot_solutions()
    plot_error_analysis()
    plot_solution_characteristics()
    plot_equation_comparison()
    
    print("All plots generated successfully!")
    print("Generated files:")
    print("- solutions_comparison.png")
    print("- error_analysis.png")
    print("- solution_characteristics.png")
    print("- equation_comparison.png")
