import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ============================
# Read Data
# ============================

def import_results(filename):
    """
    Processes a CSV file and returns a dictionary where each key is a column name
    and each value is a numpy array containing the column data.
    Handles duplicate column names by appending a suffix.
    """
    data_dict = {}
    
    with open(filename, "r") as file:
        # Read the header line to get column names
        headers = file.readline().strip().split(", ")
        
        # Handle duplicate column names
        unique_headers = []
        header_counts = {}
        
        for header in headers:
            if header in header_counts:
                header_counts[header] += 1
                unique_header = f"{header}_{header_counts[header]}"
            else:
                header_counts[header] = 0
                unique_header = header
            unique_headers.append(unique_header)
        
        # Initialize lists for each column
        for header in unique_headers:
            data_dict[header] = []
        
        # Read all data rows
        for line in file:
            row_data = line.strip().split(",")
            # Store each value in the appropriate column list
            for i, value in enumerate(row_data):
                try:
                    numeric_value = float(value)
                    data_dict[unique_headers[i]].append(numeric_value)
                except ValueError:
                    data_dict[unique_headers[i]].append(value)
    
    # Convert lists to numpy arrays
    for header in unique_headers:
        data_dict[header] = np.array(data_dict[header])
    
    return data_dict

def import_all_csv_files(directory):
    """
    Import all CSV files from the specified directory.
    
    Parameters:
        directory (str): Path to the directory containing CSV files
        
    Returns:
        dict: Dictionary where keys are filenames (without extension) and 
              values are dictionaries containing column data
    """
    all_data = {}
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {directory}")
        return all_data
    
    print(f"Found {len(csv_files)} CSV files in {directory}")
    
    for csv_file in csv_files:
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        
        try:
            print(f"Loading {filename}...")
            data = import_results(csv_file)
            all_data[filename] = data
            print(f"  ✓ Loaded {len(data)} columns with {len(next(iter(data.values())))} rows")
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {e}")
    
    return all_data

def analyze_results(data):
    """
    Analyze the results data and compute statistics.
    
    Parameters:
        data (dict): Dictionary containing column data from multiple methods
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Check if required methods exist
    methods = ['cgpops', 'no_pso', 'pso_full', 'pso_sto']
    available_methods = [m for m in methods if m in data]
    
    if not available_methods:
        print("Warning: No recognized method data found")
        return None
    N = len(data[available_methods[0]]["status"])
    
    results = {}
    
    # Extract cgpops data
    if 'cgpops' in data:
        cgpops = data["cgpops"]
        cgpops["avg_time"] = np.mean(cgpops["time"])
        avg_time_success = 0
        n_success = 0
        not_real_time = 0

        # Loop to check fake failed cgpops cases against other methods
        other_methods = available_methods[1:]  # Remove cgpops from methods
        
        for i in range(N):
            # Only process if cgpops failed (negative status)
            if cgpops["status"][i] < 0:
                cgpops_T = cgpops["T"][i]
                
                # Check against all other available methods
                for method in other_methods:
                    method_T = data[method]["T"][i]
                    method_status = data[method]["status"][i]
                    
                    # If T values are equal and the other method succeeded
                    if abs(cgpops_T - method_T) < 0.1 and method_status >= 0:
                        cgpops["status"][i] = 2
                        break  # Stop checking other methods for this case
        
        for i in range(N):
            if cgpops["time"][i] > 0.2:
                not_real_time += 1
            if cgpops["status"][i] >= 0:
                avg_time_success += cgpops["time"][i]
                n_success += 1
        
        cgpops["avg_time_success"] = avg_time_success / n_success if n_success > 0 else np.nan
        cgpops["success_rate"] = n_success / N * 100
        cgpops["not_real_time_rate"] = not_real_time / N * 100
        
        results['cgpops'] = {
            'avg_time': cgpops["avg_time"],
            'avg_time_success': cgpops["avg_time_success"],
            'success_rate': cgpops["success_rate"],
            'not_real_time_rate': cgpops["not_real_time_rate"],
            'n_success': n_success
        }

    # Extract rest of data
    comparison_methods = ['no_pso', 'pso_full', 'pso_sto']
    for col in comparison_methods:
        if col not in data:
            continue
            
        data[col]["avg_time"] = np.mean(data[col]["time"])
        n_success = 0
        T_diff = 0
        T_diff_success = 0
        time_success = 0
        not_real_time = 0
        T_diff_pso = 0
        T_diff_pso_success = 0
        time_diff = 0
        time_diff_success = 0
        
        for i in range(N):
            T = data[col]["T"][i]
            T_cgpops = data["cgpops"]["T"][i] if "cgpops" in data else 0
            T_diff += T - T_cgpops
            if data[col]["time"][i] > 0.2:
                not_real_time += 1

            

            if abs(T - T_cgpops) < 0.1 and (data["cgpops"]["status"][i] >= 0) and (data[col]["status"][i] < 0):
                data[col]["status"][i] = 2  # Mark as fake failed case

            if data[col]["status"][i] >= 0:
                time_success += data[col]["time"][i]
                T_diff_success += T - T_cgpops
                n_success += 1
            
            if col in ['pso_full', 'pso_sto'] and 'no_pso' in data:
                T_diff_pso += data[col]["T"][i] - data["no_pso"]["T"][i]
                time_diff += data[col]["time"][i] - data["no_pso"]["time"][i]
                if data[col]["status"][i] >= 0:
                    T_diff_pso_success += data[col]["T"][i] - data["no_pso"]["T"][i]
                    time_diff_success += data[col]["time"][i] - data["no_pso"]["time"][i]

        data[col]["success_rate"] = n_success / N * 100
        data[col]["avg_T_diff"] = T_diff / N
        data[col]["avg_T_diff_success"] = T_diff_success / n_success if n_success > 0 else np.nan
        data[col]["avg_time_success"] = time_success / n_success if n_success > 0 else np.nan
        data[col]["not_real_time_rate"] = not_real_time / N * 100
        
        results[col] = {
            'avg_time': data[col]["avg_time"],
            'avg_time_success': data[col]["avg_time_success"],
            'success_rate': data[col]["success_rate"],
            'not_real_time_rate': data[col]["not_real_time_rate"],
            'avg_T_diff': data[col]["avg_T_diff"],
            'avg_T_diff_success': data[col]["avg_T_diff_success"],
            'n_success': n_success
        }
        
        if col in ['pso_full', 'pso_sto']:
            if "time_on_pso" in data[col] and "time_on_solver" in data[col]:
                data[col]["avg_time_on_pso"] = np.mean(data[col]["time_on_pso"])
                data[col]["avg_time_on_solver"] = np.mean(data[col]["time_on_solver"])
                results[col]["avg_time_on_pso"] = data[col]["avg_time_on_pso"]
                results[col]["avg_time_on_solver"] = data[col]["avg_time_on_solver"]
            
            if 'no_pso' in data:
                data[col]["avg_T_diff_pso"] = T_diff_pso / N
                data[col]["avg_T_diff_pso_success"] = T_diff_pso_success / n_success if n_success > 0 else np.nan
                data[col]["avg_time_diff"] = time_diff / N
                data[col]["avg_time_diff_success"] = time_diff_success / n_success if n_success > 0 else np.nan
                
                results[col]["avg_T_diff_pso"] = data[col]["avg_T_diff_pso"]
                results[col]["avg_T_diff_pso_success"] = data[col]["avg_T_diff_pso_success"]
                results[col]["avg_time_diff"] = data[col]["avg_time_diff"]
                results[col]["avg_time_diff_success"] = data[col]["avg_time_diff_success"]

    # Print statistics
    print(f"\n{'='*60}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total Simulations: {N}\n")
    
    # Print CGPOPS results
    if 'cgpops' in results:
        print(f"CGPOPS:")
        print(f"  Success Rate: {results['cgpops']['success_rate']:.2f}%")
        print(f"  Average Time: {results['cgpops']['avg_time']:.4f} s")
        print(f"  Average Time (success only): {results['cgpops']['avg_time_success']:.4f} s")
        print(f"  Not Real-Time Rate: {results['cgpops']['not_real_time_rate']:.2f}%")
        print()
    
    # Print other methods
    for method in ['no_pso', 'pso_full', 'pso_sto']:
        if method not in results:
            continue
        
        print(f"{method.upper().replace('_', ' ')}:")
        print(f"  Success Rate: {results[method]['success_rate']:.2f}%")
        print(f"  Average Time: {results[method]['avg_time']:.4f} s")
        print(f"  Average Time (success only): {results[method]['avg_time_success']:.4f} s")
        print(f"  Not Real-Time Rate: {results[method]['not_real_time_rate']:.2f}%")
        
        if 'cgpops' in results:
            print(f"  Average T Difference vs CGPOPS: {results[method]['avg_T_diff']:.4f} s")
            print(f"  Average T Difference vs CGPOPS (success only): {results[method]['avg_T_diff_success']:.4f} s")
        
        if method in ['pso_full', 'pso_sto']:
            if 'avg_time_on_pso' in results[method]:
                print(f"  Average Time on PSO: {results[method]['avg_time_on_pso']:.4f} s")
                print(f"  Average Time on Solver: {results[method]['avg_time_on_solver']:.4f} s")
            
            if 'no_pso' in results:
                print(f"  Average T Difference vs NO_PSO: {results[method]['avg_T_diff_pso']:.4f} s")
                print(f"  Average T Difference vs NO_PSO (success only): {results[method]['avg_T_diff_pso_success']:.4f} s")
                print(f"  Average Time Difference vs NO_PSO: {results[method]['avg_time_diff']:.4f} s")
                print(f"  Average Time Difference vs NO_PSO (success only): {results[method]['avg_time_diff_success']:.4f} s")
        
        print()
    
    print(f"{'='*60}\n")
    
    results['total_sims'] = N
    return results

def plot_data(results, save_name):
    """
    Creates comprehensive bar graphs comparing different methods.
    
    Parameters:
        results (dict): Dictionary containing analysis results
        save_name (str): The name of the plot when saved.
    """
    methods = [m for m in ['cgpops', 'no_pso', 'pso_full', 'pso_sto'] if m in results and m != 'total_sims']
    
    if len(methods) < 2:
        print("Warning: Not enough methods to compare")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Method Performance Comparison', fontsize=16, fontweight='bold')
    
    # Define colors for each method
    colors = {
        'cgpops': '#FF6B6B',
        'no_pso': '#4ECDC4',
        'pso_full': '#45B7D1',
        'pso_sto': '#96CEB4'
    }
    
    method_labels = {
        'cgpops': 'CGPOPS',
        'no_pso': 'No PSO',
        'pso_full': 'PSO Full',
        'pso_sto': 'PSO STO'
    }
    
    # Plot 1: Success Rate
    ax1 = axes[0, 0]
    success_rates = [results[m]['success_rate'] for m in methods]
    bars1 = ax1.bar(range(len(methods)), success_rates, color=[colors[m] for m in methods])
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Success Rate Comparison')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([method_labels[m] for m in methods])
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Average Time
    ax2 = axes[0, 1]
    avg_times = [results[m]['avg_time'] for m in methods]
    bars2 = ax2.bar(range(len(methods)), avg_times, color=[colors[m] for m in methods])
    ax2.set_ylabel('Time (s)', fontweight='bold')
    ax2.set_title('Average Computation Time')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([method_labels[m] for m in methods])
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Not Real-Time Rate
    ax3 = axes[1, 0]
    not_rt_rates = [results[m]['not_real_time_rate'] for m in methods]
    bars3 = ax3.bar(range(len(methods)), not_rt_rates, color=[colors[m] for m in methods])
    ax3.set_ylabel('Not Real-Time Rate (%)', fontweight='bold')
    ax3.set_title('Not Real-Time Rate (> 0.2s)')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([method_labels[m] for m in methods])
    ax3.set_ylim(0, max(not_rt_rates) * 1.2)
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, not_rt_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Time breakdown for PSO methods
    ax4 = axes[1, 1]
    pso_methods = [m for m in methods if m in ['pso_full', 'pso_sto']]
    
    if pso_methods and all('avg_time_on_pso' in results[m] for m in pso_methods):
        x = np.arange(len(pso_methods))
        width = 0.35
        
        pso_times = [results[m]['avg_time_on_pso'] for m in pso_methods]
        solver_times = [results[m]['avg_time_on_solver'] for m in pso_methods]
        
        bars4a = ax4.bar(x - width/2, pso_times, width, label='PSO Time', color='#9B59B6')
        bars4b = ax4.bar(x + width/2, solver_times, width, label='Solver Time', color='#E74C3C')
        
        ax4.set_ylabel('Time (s)', fontweight='bold')
        ax4.set_title('Time Breakdown for PSO Methods')
        ax4.set_xticks(x)
        ax4.set_xticklabels([method_labels[m] for m in pso_methods])
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars4a, pso_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}s', ha='center', va='bottom', fontsize=9)
        
        for bar, val in zip(bars4b, solver_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}s', ha='center', va='bottom', fontsize=9)
    else:
        # Plot T difference if no PSO time breakdown
        if 'cgpops' in results:
            comparison_methods = [m for m in methods if m != 'cgpops' and 'avg_T_diff' in results[m]]
            if comparison_methods:
                T_diffs = [results[m]['avg_T_diff'] for m in comparison_methods]
                bars4 = ax4.bar(range(len(comparison_methods)), T_diffs, 
                              color=[colors[m] for m in comparison_methods])
                ax4.set_ylabel('T Difference (s)', fontweight='bold')
                ax4.set_title('Average T Difference vs CGPOPS')
                ax4.set_xticks(range(len(comparison_methods)))
                ax4.set_xticklabels([method_labels[m] for m in comparison_methods])
                ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
                ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                for bar, val in zip(bars4, T_diffs):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.4f}s', ha='center', 
                            va='bottom' if val >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_name, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_name}")
    plt.show()

# ============================
# Execution
# ============================

if __name__ == "__main__":
    # ------------------------------
    # Determine the correct path
    # ------------------------------
    possible_paths = [
        "../../output/mcs/",
        "../output/mcs/",
        "output/mcs/"
    ]
    
    mcs_directory = None
    for path in possible_paths:
        if os.path.exists(path):
            mcs_directory = path
            break
    
    if mcs_directory is None:
        print("Error: Could not find output/mcs/ directory")
        exit(1)
    
    print(f"Using directory: {mcs_directory}")
    
    # ------------------------------
    # Import all CSV files
    # ------------------------------
    all_data = import_all_csv_files(mcs_directory)
    
    if not all_data:
        print("Error: No data loaded")
        exit(1)
    
    # ------------------------------
    # Process each file
    # ------------------------------
    #if "pso_sto_tuning" in all_data or "pso_full_tuning" in all_data:
        #tune_pso(all_data)
    if "cgpops" in all_data:        
        summary_data = analyze_results(all_data)
        if summary_data:
            # Generate plot
            output_dir = os.path.dirname(mcs_directory.rstrip('/'))
            save_name = os.path.join(output_dir, f"mcs_analysis.pdf")
            plot_data(summary_data, save_name)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
