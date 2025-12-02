import numpy as np
import matplotlib.pyplot as plt
import os
import glob

FONTSIZE = 14
scenarios = ['rest', 'track', 'asym']
scenario_labels = ['A', 'B', 'C']
colors = ['#5E60CE', '#E63946', '#06FFA5']  # Purple, Red, Mint
colors_all = ['#87CEEB', '#87CEEB', '#87CEEB']  # Sky Blue
colors_success = ['#FF6347', '#FF6347', '#FF6347']  # Tomato Red
my_blue = '#87CEEB' 
my_red = '#FF6347'

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

def analyze_results(data_rest, data_track, data_asym):
    """
    Analyze the results data and compute statistics for three different scenarios.
    
    Parameters:
        data_rest (dict): Dictionary containing column data from multiple methods for rest-to-rest
        data_track (dict): Dictionary containing column data from multiple methods for track-to-track
        data_asym (dict): Dictionary containing column data from multiple methods for asymmetric
        
    Returns:
        dict: Dictionary containing analysis results for all three scenarios
    """
    import numpy as np
    
    # Check if required methods exist
    methods = ['cgpops', 'no_pso', 'pso_full', 'pso_sto', 'cgpops_1']
    
    all_results = {}
    scenarios = {
        'rest': data_rest,
        'track': data_track,
        'asym': data_asym
    }
    
    for scenario_name, data in scenarios.items():
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"{'='*60}")
        
        N = len(data[list(data.keys())[0]]["status"])
        results = {}
        
        # Build an "optimal" solution per sample: choose the smallest T among available methods
        optimal = {
            'T': np.full(N, np.inf),
            'status': np.full(N, -1)
        }
        
        # Initialize with cgpops if available
        if 'cgpops' in data:
            optimal['T'] = data['cgpops']["T"].copy()
            optimal['status'] = data['cgpops']["status"].copy()
        
        # Update optimal with other methods
        for m in methods:
            if m == 'cgpops' or m not in data:
                continue
            
            Tm = data[m]["T"]
            stm = data[m]["status"]
            for i in range(N):
                # only consider valid solutions
                if stm[i] >= 0:
                    if (Tm[i] < optimal['T'][i] and optimal['status'][i] >= 0) or optimal['status'][i] < 0:
                        optimal['T'][i] = Tm[i]
                        optimal['status'][i] = int(stm[i])
        
        # Extract cgpops data
        if 'cgpops' in data:
            cgpops = data["cgpops"]
            cgpops["avg_time"] = np.mean(cgpops["time"])
            cgpops["median_time"] = np.median(cgpops["time"])
            avg_time_success = 0
            n_success = 0
            not_real_time = 0
            
            # For box plots - collect successful cases
            time_success_array = []
            T_success_array = []

            for i in range(N):
                T_opt = optimal["T"][i]
                T = cgpops["T"][i]
                if cgpops["time"][i] > 0.2:
                    not_real_time += 1
                if abs(T - T_opt)/T_opt*100 < 0.1 and (optimal["status"][i] >= 0) and (cgpops["status"][i] < 0):
                    cgpops["status"][i] = 2  # Mark as fake failed case
                if cgpops["status"][i] >= 0:
                    avg_time_success += cgpops["time"][i]
                    time_success_array.append(cgpops["time"][i])
                    T_success_array.append(cgpops["T"][i])
                    n_success += 1
            
            cgpops["avg_time_success"] = avg_time_success / n_success if n_success > 0 else np.nan
            cgpops["median_time_success"] = np.median(time_success_array) if n_success > 0 else np.nan
            cgpops["success_rate"] = n_success / N * 100
            cgpops["not_real_time_rate"] = not_real_time / N * 100
            
            results['cgpops'] = {
                'avg_time': cgpops["avg_time"],
                'median_time': cgpops["median_time"],
                'avg_time_success': cgpops["avg_time_success"],
                'median_time_success': cgpops["median_time_success"],
                'success_rate': cgpops["success_rate"],
                'not_real_time_rate': cgpops["not_real_time_rate"],
                'n_success': n_success,
                'time_all': cgpops["time"].copy(),
                'time_success': np.array(time_success_array),
                'T_all': cgpops["T"].copy(),
                'T_success': np.array(T_success_array),
                'status_all': cgpops["status"].copy()
            }
        
        # Handle cgpops_1 if it exists (merge with cgpops)
        if 'cgpops_1' in data and 'cgpops' in data:
            cgpops_1 = data["cgpops_1"]
            avg_time_success = 0
            n_success = 0
            not_real_time = 0
            
            # For box plots
            time_success_array = []
            T_success_array = []

            for i in range(N):
                T_opt = optimal["T"][i]
                T = cgpops_1["T"][i]
                if abs(T - T_opt)/T_opt*100 < 0.1 and (optimal["status"][i] >= 0) and (cgpops_1["status"][i] < 0):
                    cgpops_1["status"][i] = 2  # Mark as fake failed case
                if (cgpops["T"][i] > cgpops_1["T"][i] and cgpops_1["status"][i] >= 0 and cgpops["status"][i]  >= 0) or (cgpops["status"][i] < 0 and cgpops_1["status"][i] >= 0):
                    cgpops["T"][i] = cgpops_1["T"][i]
                    cgpops["status"][i] = cgpops_1["status"][i]
                    cgpops["time"][i] = cgpops_1["time"][i]
                if cgpops["status"][i] >= 0:
                    avg_time_success += cgpops["time"][i]
                    time_success_array.append(cgpops["time"][i])
                    T_success_array.append(cgpops["T"][i])
                    n_success += 1
                if cgpops["time"][i] > 0.2:
                    not_real_time += 1

            cgpops["avg_time"] = np.mean(cgpops["time"])
            cgpops["median_time"] = np.median(cgpops["time"])
            cgpops["avg_time_success"] = avg_time_success / n_success if n_success > 0 else np.nan
            cgpops["median_time_success"] = np.median(time_success_array) if n_success > 0 else np.nan
            cgpops["success_rate"] = n_success / N * 100
            cgpops["not_real_time_rate"] = not_real_time / N * 100
            
            results['cgpops'] = {
                'avg_time': cgpops["avg_time"],
                'median_time': cgpops["median_time"],
                'avg_time_success': cgpops["avg_time_success"],
                'median_time_success': cgpops["median_time_success"],
                'success_rate': cgpops["success_rate"],
                'not_real_time_rate': cgpops["not_real_time_rate"],
                'n_success': n_success,
                'time_all': cgpops["time"].copy(),
                'time_success': np.array(time_success_array),
                'T_all': cgpops["T"].copy(),
                'T_success': np.array(T_success_array),
                'status_all': cgpops["status"].copy()
            }

        # Extract rest of data
        comparison_methods = ['no_pso', 'pso_full', 'pso_sto']
        for col in comparison_methods:
            if col not in data:
                continue
                
            data[col]["avg_time"] = np.mean(data[col]["time"])
            data[col]["median_time"] = np.median(data[col]["time"])
            n_success = 0
            n_success_pso = 0
            T_diff_fail = 0
            T_diff_success = 0
            time_success = 0
            not_real_time = 0
            T_diff_pso = 0
            time_diff = 0
            
            # For box plots
            time_success_array = []
            T_success_array = []
            T_diff_success_array = []
            T_diff_fail_array = []
            T_diff_pso_array = []
            time_diff_array = []
            
            for i in range(N):
                T = data[col]["T"][i]
                T_opt = optimal["T"][i]
                if data[col]["time"][i] > 0.2:
                    not_real_time += 1

                if abs(T - T_opt)/T_opt*100 < 0.1 and (optimal["status"][i] >= 0) and (data[col]["status"][i] < 0):
                    data[col]["status"][i] = 2  # Mark as fake failed case

                if data[col]["status"][i] >= 0:
                    time_success += data[col]["time"][i]
                    time_success_array.append(data[col]["time"][i])
                    T_success_array.append(data[col]["T"][i])
                    T_diff_pct = abs(T - T_opt)/T_opt*100
                    T_diff_success += T_diff_pct
                    T_diff_success_array.append(T_diff_pct)
                    n_success += 1
                elif optimal["status"][i] >= 0:
                    T_diff_pct = abs(T - T_opt)/T_opt*100
                    T_diff_fail += T_diff_pct
                    T_diff_fail_array.append(T_diff_pct)

                
                if col in ['pso_full', 'pso_sto'] and 'no_pso' in data:
                    if data[col]["status"][i] >= 0 and data["no_pso"]["status"][i] >= 0:
                        T_diff_pso_pct = (data[col]["T"][i] - data["no_pso"]["T"][i])/data["no_pso"]["T"][i] * 100
                        T_diff_pso += T_diff_pso_pct
                        T_diff_pso_array.append(T_diff_pso_pct)
                        time_diff_val = data[col]["time"][i] - data["no_pso"]["time"][i]
                        time_diff += time_diff_val
                        time_diff_array.append(time_diff_val)
                        n_success_pso += 1

            data[col]["success_rate"] = n_success / N * 100
            data[col]["rel_success_rate"] = n_success / results['cgpops']['n_success'] * 100 if 'cgpops' in results else np.nan
            data[col]["avg_T_diff_fail"] = T_diff_fail / (N - n_success) if (N - n_success) > 0 else np.nan
            data[col]["avg_T_diff_success"] = T_diff_success / n_success if n_success > 0 else np.nan
            data[col]["avg_time_success"] = time_success / n_success if n_success > 0 else np.nan
            data[col]["median_time_success"] = np.median(time_success_array) if n_success > 0 else np.nan
            data[col]["median_T_diff_success"] = np.median(T_diff_success_array) if n_success > 0 else np.nan
            data[col]["median_T_diff_fail"] = np.median(T_diff_fail_array) if (N - n_success) > 0 else np.nan
            data[col]["not_real_time_rate"] = not_real_time / N * 100
            
            results[col] = {
                'avg_time': data[col]["avg_time"],
                'median_time': data[col]["median_time"],
                'avg_time_success': data[col]["avg_time_success"],
                'median_time_success': data[col]["median_time_success"],
                'success_rate': data[col]["success_rate"],
                'rel_success_rate': data[col]["rel_success_rate"],
                'not_real_time_rate': data[col]["not_real_time_rate"],
                'avg_T_diff_fail': data[col]["avg_T_diff_fail"],
                'median_T_diff_fail': data[col]["median_T_diff_fail"],
                'avg_T_diff_success': data[col]["avg_T_diff_success"],
                'median_T_diff_success': data[col]["median_T_diff_success"],
                'n_success': n_success,
                'time_all': data[col]["time"].copy(),
                'time_success': np.array(time_success_array),
                'T_all': data[col]["T"].copy(),
                'T_success': np.array(T_success_array),
                'T_diff_success': np.array(T_diff_success_array),
                'T_diff_fail': np.array(T_diff_fail_array),
                'status_all': data[col]["status"].copy()
            }
            
            if col in ['pso_full', 'pso_sto']:
                if "time_on_pso" in data[col] and "time_on_solver" in data[col]:
                    data[col]["avg_time_on_pso"] = np.mean(data[col]["time_on_pso"])
                    data[col]["avg_time_on_solver"] = np.mean(data[col]["time_on_solver"])
                    results[col]["avg_time_on_pso"] = data[col]["avg_time_on_pso"]
                    results[col]["avg_time_on_solver"] = data[col]["avg_time_on_solver"]
                    data[col]["median_time_on_pso"] = np.median(data[col]["time_on_pso"])
                    data[col]["median_time_on_solver"] = np.median(data[col]["time_on_solver"])
                    results[col]["median_time_on_pso"] = data[col]["median_time_on_pso"]
                    results[col]["median_time_on_solver"] = data[col]["median_time_on_solver"]  
                    # Raw data for box plots
                    results[col]["time_on_pso_all"] = data[col]["time_on_pso"].copy()
                    results[col]["time_on_solver_all"] = data[col]["time_on_solver"].copy()
                
                if 'no_pso' in data:
                    data[col]["avg_T_diff_pso"] = T_diff_pso / n_success_pso if n_success_pso > 0 else np.nan
                    data[col]["avg_time_diff"] = time_diff / n_success_pso if n_success_pso > 0 else np.nan
                    data[col]["median_T_diff_pso"] = np.median(T_diff_pso_array) if n_success_pso > 0 else np.nan
                    data[col]["median_time_diff"] = np.median(time_diff_array) if n_success_pso > 0 else np.nan
                    results[col]["median_T_diff_pso"] = data[col]["median_T_diff_pso"]
                    results[col]["median_time_diff"] = data[col]["median_time_diff"]
                    results[col]["avg_T_diff_pso"] = data[col]["avg_T_diff_pso"]
                    results[col]["avg_time_diff"] = data[col]["avg_time_diff"]                    # Raw data for box plots
                    results[col]["T_diff_pso"] = np.array(T_diff_pso_array)
                    results[col]["time_diff"] = np.array(time_diff_array)

        # Print statistics for this scenario
        print(f"Total Simulations: {N}\n")
        
        # Print CGPOPS results
        if 'cgpops' in results:
            print(f"CGPOPS:")
            print(f"  Success Rate: {results['cgpops']['success_rate']:.2f}%")
            print(f"  Average Time: {results['cgpops']['avg_time']:.4f} s")
            print(f"  Median Time: {results['cgpops']['median_time']:.4f} s")
            print(f"  Average Time (success only): {results['cgpops']['avg_time_success']:.4f} s")
            print(f"  Median Time (success only): {results['cgpops']['median_time_success']:.4f} s")
            print(f"  Not Real-Time Rate: {results['cgpops']['not_real_time_rate']:.2f}%")
            print()
        
        # Print other methods
        for method in ['no_pso', 'pso_full', 'pso_sto']:
            if method not in results:
                continue
            
            print(f"{method.upper().replace('_', ' ')}:")
            print(f"  Success Rate: {results[method]['success_rate']:.2f}%")
            print(f"  Relative Success Rate vs CGPOPS: {results[method]['rel_success_rate']:.2f}%")
            print(f"  Average Time: {results[method]['avg_time']:.4f} s")
            print(f"  Median Time: {results[method]['median_time']:.4f} s")
            print(f"  Average Time (success only): {results[method]['avg_time_success']:.4f} s")
            print(f"  Median Time (success only): {results[method]['median_time_success']:.4f} s")
            print(f"  Not Real-Time Rate: {results[method]['not_real_time_rate']:.2f}%")
            
            if 'cgpops' in results:
                print(f"  Average T Difference vs CGPOPS (failures only): {results[method]['avg_T_diff_fail']:.4f}%")
                print(f"  Median T Difference vs CGPOPS (failures only): {results[method]['median_T_diff_fail']:.4f}%")
                print(f"  Average T Difference vs CGPOPS (success only): {results[method]['avg_T_diff_success']:.4f}%")
                print(f"  Median T Difference vs CGPOPS (success only): {results[method]['median_T_diff_success']:.4f}%")
            
            if method in ['pso_full', 'pso_sto']:
                if 'avg_time_on_pso' in results[method]:
                    print(f"  Average Time on PSO: {results[method]['avg_time_on_pso']:.4f} s")
                    print(f"  Median Time on PSO: {results[method]['median_time_on_pso']:.4f} s")
                    print(f"  Average Time on Solver: {results[method]['avg_time_on_solver']:.4f} s")
                    print(f"  Median Time on Solver: {results[method]['median_time_on_solver']:.4f} s")
                
                if 'no_pso' in results:
                    print(f"  Average T Difference vs NO_PSO: {results[method]['avg_T_diff_pso']:.4f}%")
                    print(f"  Median T Difference vs NO_PSO: {results[method]['median_T_diff_pso']:.4f}%")
                    print(f"  Average Time Difference vs NO_PSO: {results[method]['avg_time_diff']:.4f} s")
                    print(f"  Median Time Difference vs NO_PSO: {results[method]['median_time_diff']:.4f} s")
            
            print()
        
        results['total_sims'] = N
        all_results[scenario_name] = results
    
    print(f"{'='*60}\n")
    
    return all_results

def plot_computation_time_no_pso(all_results, save_dir="output/mcs"):
    """
    Plot combined box plot with computation times for NO_PSO.
    Each scenario has two columns (all cases and success cases).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots()
    
    # Prepare positions for grouped bars
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    width = 0.35
    
    # Create positions for each box
    positions_all = [i - width/2 for i in x]
    positions_success = [i + width/2 for i in x]
    
    data_all = []
    data_success = []
    for scenario in scenarios:
        if 'no_pso' in all_results[scenario]:
            data_all.append(all_results[scenario]['no_pso']['time_all'])
            if len(all_results[scenario]['no_pso']['time_success']) > 0:
                data_success.append(all_results[scenario]['no_pso']['time_success'])
            else:
                data_success.append([])
        else:
            data_all.append([])
            data_success.append([])
    
    # Plot all cases
    bp1 = ax.boxplot(data_all, positions=positions_all, widths=width*0.8,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp1['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot success cases
    bp2 = ax.boxplot(data_success, positions=positions_success, widths=width*0.8,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp2['boxes'], colors_success):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Computation Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('No PSO: Computation Time', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='All Cases'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Success Cases'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'computation_time_no_pso.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_computation_time_cgpops(all_results, save_dir="output/mcs"):
    """
    Plot combined box plot with computation times for CGPOPS.
    Each scenario has two columns (all cases and success cases).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots()
    
    # Prepare positions for grouped bars
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    width = 0.35
    
    data_all = []
    data_success = []
    for scenario in scenarios:
        if 'cgpops' in all_results[scenario]:
            data_all.append(all_results[scenario]['cgpops']['time_all'])
            if len(all_results[scenario]['cgpops']['time_success']) > 0:
                data_success.append(all_results[scenario]['cgpops']['time_success'])
            else:
                data_success.append([])
        else:
            data_all.append([])
            data_success.append([])
    
    # Create positions for each box
    positions_all = [i - width/2 for i in x]
    positions_success = [i + width/2 for i in x]
    
    # Plot all cases
    bp1 = ax.boxplot(data_all, positions=positions_all, widths=width*0.8, 
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp1['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot success cases
    bp2 = ax.boxplot(data_success, positions=positions_success, widths=width*0.8,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp2['boxes'], colors_success):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Computation Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('CGPOPS: Computation Time', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='All Cases'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Success Cases'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE, loc='upper left')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'computation_time_cgpops.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_cgpops_success_rate(all_results, save_dir="output/mcs"):
    """
    Plot bar graph with success rate of CGPOPS for all 3 scenarios.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    success_rates = []
    for scenario in scenarios:
        if 'cgpops' in all_results[scenario]:
            success_rates.append(all_results[scenario]['cgpops']['success_rate'])
        else:
            success_rates.append(0)
    
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(scenarios)), success_rates, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('CGPOPS Success Rate Across Scenarios', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenario_labels, fontsize=FONTSIZE)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)


    # Customize colors
    for patch, color in zip(bars, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=FONTSIZE, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cgpops_success_rate.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_no_pso_relative_success_rate(all_results, save_dir="output/mcs"):
    """
    Plot bar graph with relative success rate of no_pso for all 3 scenarios.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    rel_success_rates = []
    for scenario in scenarios:
        if 'no_pso' in all_results[scenario]:
            rel_success_rates.append(all_results[scenario]['no_pso']['rel_success_rate'])
        else:
            rel_success_rates.append(0)
    
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(scenarios)), rel_success_rates, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Relative Success Rate (%)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('No PSO Relative Success Rate', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenario_labels, fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% (Benchmark algorithm)')
    ax.legend(fontsize=FONTSIZE, bbox_to_anchor=(0.99, 0.01), loc='lower right')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)

    # Customize colors
    for patch, color in zip(bars, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, rel_success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - 8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=FONTSIZE, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'no_pso_relative_success_rate.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_no_pso_T_diff_boxplot(all_results, save_dir="output/mcs"):
    """
    Plot box plot with T_diff_success of no_pso for all 3 scenarios.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    data_to_plot = []
    for scenario in scenarios:
        if 'no_pso' in all_results[scenario] and len(all_results[scenario]['no_pso']['T_diff_fail']) > 0:
            data_to_plot.append(all_results[scenario]['no_pso']['T_diff_fail'])
        else:
            data_to_plot.append([])
    
    fig, ax = plt.subplots()
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=scenario_labels, patch_artist=True,
                    widths=0.6, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    # Customize colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('T Difference vs Optimal (%)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('No PSO: T Difference (Failure Cases Only)', fontweight='bold', fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'no_pso_T_diff_boxplot.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_pso_relative_success_rate(all_results, save_dir="output/mcs"):
    """
    Plot bar graph with relative success rate of no_pso, pso_full, and pso_sto for all 3 scenarios.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    methods = ['no_pso', 'pso_full', 'pso_sto']
    method_labels = ['No PSO', 'PSO Full', 'PSO STO']
    
    fig, ax = plt.subplots()
    
    # Prepare positions for grouped bars
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    width = 0.25
    
    # Create positions for each method
    positions = [x - width, x, x + width]
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        rel_success_rates = []
        for scenario in scenarios:
            if method in all_results[scenario]:
                rel_success_rates.append(all_results[scenario][method]['rel_success_rate'])
            else:
                rel_success_rates.append(0)
        
        bars = ax.bar(positions[i], rel_success_rates, width, label=label, 
                     color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, rel_success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Relative Success Rate (%)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO Methods Relative Success Rate vs CGPOPS', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% (Benchmark algorithm)')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pso_methods_relative_success_rate.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_full_T_diff_fail_boxplot(all_results, save_dir="output/mcs"):
    """
    Plot box plot with T_diff_fail for pso_full for all 3 scenarios.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    data_to_plot = []
    for scenario in scenarios:
        if 'pso_full' in all_results[scenario] and len(all_results[scenario]['pso_full']['T_diff_fail']) > 0:
            data_to_plot.append(all_results[scenario]['pso_full']['T_diff_fail'])
        else:
            data_to_plot.append([])
    
    fig, ax = plt.subplots()
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=scenario_labels, patch_artist=True,
                    widths=0.6, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    # Customize colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('T Difference vs Optimal (%)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO Full: T Difference (Failure Cases Only)', fontweight='bold', fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pso_full_T_diff_fail_boxplot.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_sto_T_diff_fail_boxplot(all_results, save_dir="output/mcs"):
    """
    Plot box plot with T_diff_fail for pso_sto for all 3 scenarios.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    data_to_plot = []
    for scenario in scenarios:
        if 'pso_sto' in all_results[scenario] and len(all_results[scenario]['pso_sto']['T_diff_fail']) > 0:
            data_to_plot.append(all_results[scenario]['pso_sto']['T_diff_fail'])
        else:
            data_to_plot.append([])
    
    fig, ax = plt.subplots()
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=scenario_labels, patch_artist=True,
                    widths=0.6, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    # Customize colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('T Difference vs Optimal (%)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO STO: T Difference (Failure Cases Only)', fontweight='bold', fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pso_sto_T_diff_fail_boxplot.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_pso_full_time_breakdown_AB(all_results, save_dir="output/mcs"):
    """
    Plot box plot showing PSO time and solver time for pso_full for scenarios A and B.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    scenarios_subset = ['rest', 'track']
    scenario_labels_subset = ['A', 'B']
    colors_subset = [my_blue, my_red]  # Purple, Red
    
    fig, ax = plt.subplots()
    
    # Prepare positions for grouped bars
    n_scenarios = len(scenarios_subset)
    x = np.arange(n_scenarios)
    width = 0.35
    
    # Create positions for each time type
    positions_pso = [i - width/2 for i in x]
    positions_solver = [i + width/2 for i in x]
    
    data_pso = []
    data_solver = []
    for scenario in scenarios_subset:
        if 'pso_full' in all_results[scenario] and 'time_on_pso_all' in all_results[scenario]['pso_full']:
            data_pso.append(all_results[scenario]['pso_full']['time_on_pso_all'])
            data_solver.append(all_results[scenario]['pso_full']['time_on_solver_all'])
        else:
            data_pso.append([])
            data_solver.append([])
    
    # Plot PSO time
    bp1 = ax.boxplot(data_pso, positions=positions_pso, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch in bp1['boxes']:
        patch.set_facecolor(my_blue)  # Purple
        patch.set_alpha(0.7)
    
    # Plot Solver time
    bp2 = ax.boxplot(data_solver, positions=positions_solver, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch in bp2['boxes']:
        patch.set_facecolor(my_red)  # Red
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO Full: Time Breakdown (Scenarios A & B)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels_subset)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='PSO Time'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Solver Time')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pso_full_time_breakdown_AB.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_pso_sto_time_breakdown_AB(all_results, save_dir="output/mcs"):
    """
    Plot box plot showing PSO time and solver time for pso_sto for scenarios A and B.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    scenarios_subset = ['rest', 'track']
    scenario_labels_subset = ['A', 'B']
    colors_subset = [my_blue, my_red]  # Purple, Red
    
    fig, ax = plt.subplots()
    
    # Prepare positions for grouped bars
    n_scenarios = len(scenarios_subset)
    x = np.arange(n_scenarios)
    width = 0.35
    
    # Create positions for each time type
    positions_pso = [i - width/2 for i in x]
    positions_solver = [i + width/2 for i in x]
    
    data_pso = []
    data_solver = []
    for scenario in scenarios_subset:
        if 'pso_sto' in all_results[scenario] and 'time_on_pso_all' in all_results[scenario]['pso_sto']:
            data_pso.append(all_results[scenario]['pso_sto']['time_on_pso_all'])
            data_solver.append(all_results[scenario]['pso_sto']['time_on_solver_all'])
        else:
            data_pso.append([])
            data_solver.append([])
    
    # Plot PSO time
    bp1 = ax.boxplot(data_pso, positions=positions_pso, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch in bp1['boxes']:
        patch.set_facecolor(my_blue)  # Purple
        patch.set_alpha(0.7)
    
    # Plot Solver time
    bp2 = ax.boxplot(data_solver, positions=positions_solver, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch in bp2['boxes']:
        patch.set_facecolor(my_red)  # Red
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO STO: Time Breakdown (Scenarios A & B)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels_subset)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='PSO Time'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Solver Time')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE, bbox_to_anchor=(0.01, 0.95), loc='upper left')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pso_sto_time_breakdown_AB.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_pso_full_time_breakdown_C(all_results, save_dir="output/mcs"):
    """
    Plot box plot showing PSO time and solver time for pso_full for scenario C.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    scenario = 'asym'
    scenario_label = 'C'
    
    fig, ax = plt.subplots()
    
    # Prepare data
    data_pso = []
    data_solver = []
    if 'pso_full' in all_results[scenario] and 'time_on_pso_all' in all_results[scenario]['pso_full']:
        data_pso = [all_results[scenario]['pso_full']['time_on_pso_all']]
        data_solver = [all_results[scenario]['pso_full']['time_on_solver_all']]
    
    if not data_pso:
        print(f"Warning: No data available for PSO Full scenario C")
        return
    
    # Create positions for each time type
    x = [0]
    width = 0.35
    positions_pso = [-width/2]
    positions_solver = [width/2]
    
    # Plot PSO time
    bp1 = ax.boxplot(data_pso, positions=positions_pso, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch in bp1['boxes']:
        patch.set_facecolor(my_blue)  # Purple
        patch.set_alpha(0.7)
    
    # Plot Solver time
    bp2 = ax.boxplot(data_solver, positions=positions_solver, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch in bp2['boxes']:
        patch.set_facecolor(my_red)  # Red
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO Full: Time Breakdown (Scenario C)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_label])
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='PSO Time'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Solver Time')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pso_full_time_breakdown_C.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_pso_sto_time_breakdown_C(all_results, save_dir="output/mcs"):
    """
    Plot box plot showing PSO time and solver time for pso_sto for scenario C.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    scenario = 'asym'
    scenario_label = 'C'
    
    fig, ax = plt.subplots()
    
    # Prepare data
    data_pso = []
    data_solver = []
    if 'pso_sto' in all_results[scenario] and 'time_on_pso_all' in all_results[scenario]['pso_sto']:
        data_pso = [all_results[scenario]['pso_sto']['time_on_pso_all']]
        data_solver = [all_results[scenario]['pso_sto']['time_on_solver_all']]
    
    if not data_pso:
        print(f"Warning: No data available for PSO STO scenario C")
        return
    
    # Create positions for each time type
    x = [0]
    width = 0.35
    positions_pso = [-width/2]
    positions_solver = [width/2]
    
    # Plot PSO time
    bp1 = ax.boxplot(data_pso, positions=positions_pso, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch in bp1['boxes']:
        patch.set_facecolor(my_blue)  # Purple
        patch.set_alpha(0.7)
    
    # Plot Solver time
    bp2 = ax.boxplot(data_solver, positions=positions_solver, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch in bp2['boxes']:
        patch.set_facecolor(my_red)  # Red
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO STO: Time Breakdown (Scenario C)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_label])
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='PSO Time'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Solver Time')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pso_sto_time_breakdown_C.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_computation_time_pso_full_AB(all_results, save_dir="output/mcs"):
    """
    Plot combined box plot with computation times for PSO Full for scenarios A and B.
    Each scenario has two columns (all cases and success cases).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    scenarios_subset = ['rest', 'track']
    scenario_labels_subset = ['A', 'B']
    
    fig, ax = plt.subplots()
    
    # Prepare positions for grouped bars
    n_scenarios = len(scenarios_subset)
    x = np.arange(n_scenarios)
    width = 0.35
    
    data_all = []
    data_success = []
    for scenario in scenarios_subset:
        if 'pso_full' in all_results[scenario]:
            data_all.append(all_results[scenario]['pso_full']['time_all'])
            if len(all_results[scenario]['pso_full']['time_success']) > 0:
                data_success.append(all_results[scenario]['pso_full']['time_success'])
            else:
                data_success.append([])
        else:
            data_all.append([])
            data_success.append([])
    
    # Create positions for each box
    positions_all = [i - width/2 for i in x]
    positions_success = [i + width/2 for i in x]
    
    colors_all = [my_blue, my_blue]  # Sky Blue
    colors_success = [my_red, my_red]  # Tomato Red
    
    # Plot all cases
    bp1 = ax.boxplot(data_all, positions=positions_all, widths=width*0.8, 
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp1['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot success cases
    bp2 = ax.boxplot(data_success, positions=positions_success, widths=width*0.8,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp2['boxes'], colors_success):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Computation Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO Full: Computation Time (Scenarios A & B)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels_subset)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='All Cases'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Success Cases'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE, bbox_to_anchor=(0.99, 0.98), loc='upper right')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'computation_time_pso_full_AB.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_computation_time_pso_full_C(all_results, save_dir="output/mcs"):
    """
    Plot combined box plot with computation times for PSO Full for scenario C.
    Each scenario has two columns (all cases and success cases).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    scenario = 'asym'
    scenario_label = 'C'
    
    fig, ax = plt.subplots()
    
    # Prepare data
    data_all = []
    data_success = []
    if 'pso_full' in all_results[scenario]:
        data_all = [all_results[scenario]['pso_full']['time_all']]
        if len(all_results[scenario]['pso_full']['time_success']) > 0:
            data_success = [all_results[scenario]['pso_full']['time_success']]
        else:
            data_success = [[]]
    
    if not data_all or len(data_all[0]) == 0:
        print(f"Warning: No data available for PSO Full scenario C")
        return
    
    # Create positions for each box
    x = [0]
    width = 0.35
    positions_all = [-width/2]
    positions_success = [width/2]
    
    colors_all = [my_blue]  # Sky Blue
    colors_success = [my_red]  # Tomato Red
    
    # Plot all cases
    bp1 = ax.boxplot(data_all, positions=positions_all, widths=width*0.8, 
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp1['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot success cases
    if data_success and len(data_success[0]) > 0:
        bp2 = ax.boxplot(data_success, positions=positions_success, widths=width*0.8,
                          patch_artist=True, showfliers=False,
                          boxprops=dict(linewidth=1.5),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5),
                          medianprops=dict(linewidth=2, color='black'))
        
        for patch, color in zip(bp2['boxes'], colors_success):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('Computation Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO Full: Computation Time (Scenario C)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_label])
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='All Cases'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Success Cases'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE, bbox_to_anchor=(0.01, 0.50), loc='lower left')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'computation_time_pso_full_C.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_computation_time_pso_sto_AB(all_results, save_dir="output/mcs"):
    """
    Plot combined box plot with computation times for PSO STO for scenarios A and B.
    Each scenario has two columns (all cases and success cases).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    scenarios_subset = ['rest', 'track']
    scenario_labels_subset = ['A', 'B']
    
    fig, ax = plt.subplots()
    
    # Prepare positions for grouped bars
    n_scenarios = len(scenarios_subset)
    x = np.arange(n_scenarios)
    width = 0.35
    
    data_all = []
    data_success = []
    for scenario in scenarios_subset:
        if 'pso_sto' in all_results[scenario]:
            data_all.append(all_results[scenario]['pso_sto']['time_all'])
            if len(all_results[scenario]['pso_sto']['time_success']) > 0:
                data_success.append(all_results[scenario]['pso_sto']['time_success'])
            else:
                data_success.append([])
        else:
            data_all.append([])
            data_success.append([])
    
    # Create positions for each box
    positions_all = [i - width/2 for i in x]
    positions_success = [i + width/2 for i in x]
    
    colors_all = [my_blue, my_blue]  # Sky Blue
    colors_success = [my_red, my_red]  # Tomato Red
    
    # Plot all cases
    bp1 = ax.boxplot(data_all, positions=positions_all, widths=width*0.8, 
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp1['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot success cases
    bp2 = ax.boxplot(data_success, positions=positions_success, widths=width*0.8,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp2['boxes'], colors_success):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Computation Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO STO: Computation Time (Scenarios A & B)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels_subset)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='All Cases'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Success Cases'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE, bbox_to_anchor=(0.99, 0.45), loc='lower right')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'computation_time_pso_sto_AB.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_computation_time_pso_sto_C(all_results, save_dir="output/mcs"):
    """
    Plot combined box plot with computation times for PSO STO for scenario C.
    Each scenario has two columns (all cases and success cases).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    scenario = 'asym'
    scenario_label = 'C'
    
    fig, ax = plt.subplots()
    
    # Prepare data
    data_all = []
    data_success = []
    if 'pso_sto' in all_results[scenario]:
        data_all = [all_results[scenario]['pso_sto']['time_all']]
        if len(all_results[scenario]['pso_sto']['time_success']) > 0:
            data_success = [all_results[scenario]['pso_sto']['time_success']]
        else:
            data_success = [[]]
    
    if not data_all or len(data_all[0]) == 0:
        print(f"Warning: No data available for PSO STO scenario C")
        return
    
    # Create positions for each box
    x = [0]
    width = 0.35
    positions_all = [-width/2]
    positions_success = [width/2]
    
    colors_all = [my_blue]  # Sky Blue
    colors_success = [my_red]  # Tomato Red
    
    # Plot all cases
    bp1 = ax.boxplot(data_all, positions=positions_all, widths=width*0.8, 
                      patch_artist=True, showfliers=False,
                      boxprops=dict(linewidth=1.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5),
                      medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp1['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot success cases
    if data_success and len(data_success[0]) > 0:
        bp2 = ax.boxplot(data_success, positions=positions_success, widths=width*0.8,
                          patch_artist=True, showfliers=False,
                          boxprops=dict(linewidth=1.5),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5),
                          medianprops=dict(linewidth=2, color='black'))
        
        for patch, color in zip(bp2['boxes'], colors_success):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('Computation Time (s)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_title('PSO STO: Computation Time (Scenario C)', fontweight='bold', fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_label])
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ax.set_xlabel('Scenario', fontweight='bold', fontsize=FONTSIZE)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=my_blue, edgecolor='black', linewidth=1, alpha=0.7, label='All Cases'),
        Patch(facecolor=my_red, edgecolor='black', linewidth=1, alpha=0.7, label='Success Cases'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Real-time limit')
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE, bbox_to_anchor=(0.01, 0.50), loc='lower left')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'computation_time_pso_sto_C.pdf')
    plt.savefig(save_path, format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_data(all_results, save_dir="output/mcs"):
    """
    Generate all plots based on the analyzed results.
    """
    plot_cgpops_success_rate(all_results, save_dir)
    plot_no_pso_relative_success_rate(all_results, save_dir)
    plot_no_pso_T_diff_boxplot(all_results, save_dir)
    plot_computation_time_no_pso(all_results, save_dir)
    plot_computation_time_cgpops(all_results, save_dir)
    plot_pso_relative_success_rate(all_results, save_dir)
    plot_full_T_diff_fail_boxplot(all_results, save_dir)
    plot_sto_T_diff_fail_boxplot(all_results, save_dir)
    plot_pso_full_time_breakdown_AB(all_results, save_dir)
    plot_pso_sto_time_breakdown_AB(all_results, save_dir)
    plot_pso_full_time_breakdown_C(all_results, save_dir)
    plot_pso_sto_time_breakdown_C(all_results, save_dir)
    plot_computation_time_pso_full_AB(all_results, save_dir)
    plot_computation_time_pso_full_C(all_results, save_dir)
    plot_computation_time_pso_sto_AB(all_results, save_dir)
    plot_computation_time_pso_sto_C(all_results, save_dir)

# ============================
# Execution
# ============================

if __name__ == "__main__":
    # ------------------------------
    # Determine the correct path
    # ------------------------------
    possible_paths = [
        "../../../output/mcs/",
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
    rest_dir = mcs_directory + "rest/"
    track_dir = mcs_directory + "track/"
    asym_dir = mcs_directory + "asym/"
    data_rest = import_all_csv_files(rest_dir)
    data_track = import_all_csv_files(track_dir)
    data_asym = import_all_csv_files(asym_dir)
    
    if not (data_rest or data_track or data_asym):
        print("Error: No data loaded")
        exit(1)
    
    # ------------------------------
    # Process each file
    # ------------------------------
       
    summary_data = analyze_results(data_rest, data_track, data_asym)
    if summary_data:
        # Generate plot
        plot_data(summary_data, mcs_directory)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
