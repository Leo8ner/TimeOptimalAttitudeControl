import numpy as np
import matplotlib.pyplot as plt

# ============================
# Read Data
# ============================

def import_results(filename):
    """
    Processes the CSV file and returns a dictionary where each key is a column name
    and each value is a numpy array containing the column data.
    Handles duplicate column names by appending a suffix.
    """
    data_dict = {}
    
    with open(filename, "r") as file:
        # Read the header line to get column names
        headers = file.readline().strip().split(",")
        
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

def plot_data(data_dict, save_name):
    """
    Creates a bar graph with three groups: Success, Solution, and Time performance.
    
    Parameters:
        data_dict (dict): Dictionary containing analysis results
        save_name (str): The name of the plot when saved.
    """
    # Extract data from dictionary
    N = data_dict['total_sims']
    failed_both = data_dict['failed_both']
    failed_sol = data_dict['failed_sol']
    failed_pso = data_dict['failed_pso']
    pso_better = data_dict['pso_better']
    pso_worse = data_dict['pso_worse']
    pso_same = data_dict['pso_same']
    solver_faster = data_dict['solver_faster']
    solver_slower = data_dict['solver_slower']
    
    successful_sims = N - failed_sol - failed_pso + failed_both  # Adjust for double-counted failures
    
    # Calculate percentages
    # Group 1: Success rates (out of total simulations)
    success_percentages = [
        (failed_both / N) * 100,           # Failed simulations
        (failed_sol / N) * 100,         # Repeated PSO
        (failed_pso / N) * 100            # Failed PSO
    ]
    
    # Group 2: Solution quality (out of successful simulations)
    solution_percentages = [
        (pso_better / N) * 100,  # PSO better
        (pso_worse / N) * 100,   # PSO worse
        (pso_same / N) * 100     # Equal solutions
    ]
    
    # Group 3: Time performance (out of successful simulations)
    time_percentages = [
        (solver_faster / successful_sims) * 100,  # Solver faster with PSO
        (solver_slower / successful_sims) * 100   # Solver slower with PSO
    ]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define bar positions
    x_pos = np.arange(3)  # Three groups
    bar_width = 0.25
    
    # Define colors for each category
    colors_group1 = ['#FF6B6B', '#FFE66D', '#FF8E53']  # Red tones for failures
    colors_group2 = ['#4ECDC4', '#45B7D1', '#96CEB4']  # Blue-green tones for solutions
    colors_group3 = ['#9B59B6', '#E74C3C']             # Purple-red for time
    
    # Group 1: Success bars
    bars1_1 = ax.bar(x_pos[0] - bar_width, success_percentages[0], bar_width, 
                     color=colors_group1[0])
    bars1_2 = ax.bar(x_pos[0], success_percentages[1], bar_width, 
                     color=colors_group1[1])
    bars1_3 = ax.bar(x_pos[0] + bar_width, success_percentages[2], bar_width, 
                     color=colors_group1[2])
    
    # Group 2: Solution bars
    bars2_1 = ax.bar(x_pos[1] - bar_width, solution_percentages[0], bar_width, 
                     color=colors_group2[0])
    bars2_2 = ax.bar(x_pos[1], solution_percentages[1], bar_width, 
                     color=colors_group2[1])
    bars2_3 = ax.bar(x_pos[1] + bar_width, solution_percentages[2], bar_width, 
                     color=colors_group2[2])
    
    # Group 3: Time bars
    bars3_1 = ax.bar(x_pos[2] - bar_width/2, time_percentages[0], bar_width, 
                     color=colors_group3[0])
    bars3_2 = ax.bar(x_pos[2] + bar_width/2, time_percentages[1], bar_width, 
                     color=colors_group3[1])
    
    # Add percentage labels on bars at the top
    def add_percentage_labels(bars, percentages, labels):
        for bar_container, percentage, label in zip(bars, percentages, labels):
            for bar in bar_container:  # Iterate over each bar in the BarContainer
                height = bar.get_height()
                # Position percentage label at the top of the bar
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,  # Position above the bar
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black')
                
                # Check if the bar is short for vertical label placement
                if height < 10:  # Adjust this threshold as needed
                    ax.text(bar.get_x() + bar.get_width()/2., height + 4,  # Position above the bar
                            label, ha='center', va='bottom', fontsize=10, color='black')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height / 2,  # Position inside the bar
                            label, ha='center', va='center', fontsize=10, color='black', rotation=90)
    
    add_percentage_labels([bars1_1, bars1_2, bars1_3], success_percentages, ['Failed Both', 'Failed Sol', 'Failed PSO'])
    add_percentage_labels([bars2_1, bars2_2, bars2_3], solution_percentages, ['PSO Better', 'PSO Worse', 'Equal Solutions'])
    add_percentage_labels([bars3_1, bars3_2], time_percentages, ['Faster w/ PSO', 'Slower w/ PSO'])
    
    # Customize the plot
    ax.set_xlabel('Performance Categories', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('PSO Performance Analysis', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Success Rate', 'Solution Quality', 'Time Performance'])
    
    # Set y-axis to show appropriate range
    max_percentage = max(max(success_percentages), max(solution_percentages), max(time_percentages))
    ax.set_ylim(0, max_percentage + 10)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{save_name}", format="pdf", dpi=600, transparent=False, bbox_inches='tight')
    plt.show()

# ============================
# Execution
# ============================

if __name__ == "__main__":
    # ------------------------------
    # Read and Process Data
    # ------------------------------
    try:
        data = import_results("../../output/mcs/results.csv")
        save_name = "../../output/results_analysis.pdf"
    except Exception as e:
        try:
            data = import_results("../output/mcs/results.csv")
            save_name = "../output/results_analysis.pdf"
        except Exception as e:
            print("Error: Could not read results file.")
            raise e

    time_spent_in_pso = np.mean(data["pso_time"])
    time_solver_Wpso = np.mean(data["solve_time"])
    time_solver_WOpso = np.mean(data["solve_time_1"])
    total_time = np.mean(data["total_time"])
    sol_comparison = np.mean(data["sol_comparison"])
    time_comparison = np.mean(data["time_comparison"])
    failed_sol = 0
    failed_both = 0
    failed_pso = 0
    pso_better = 0
    pso_worse = 0
    pso_same = 0
    solver_faster = 0
    solver_slower = 0
    failed_sol_indices = []
    failed_pso_indices = []
    pso_failed = False
    solver_failed = False

    N = len(data["pso_time"])
    for i in range(N):
        if data["status"][i] == -3:
            failed_sol += 1
            failed_sol_indices.append(i)
            solver_failed = True
        elif data["status"][i] == -4:
            failed_pso += 1
            failed_pso_indices.append(i)
            pso_failed = True
        elif data["status"][i] == -34:
            failed_sol += 1
            failed_pso += 1
            failed_both += 1
            failed_sol_indices.append(i)
            failed_pso_indices.append(i)
            solver_failed = True
            pso_failed = True

        if (round(data["sol_comparison"][i], 3)  > 0 and not pso_failed) or (not pso_failed and solver_failed):
            pso_better += 1
        elif (round(data["sol_comparison"][i], 3) < 0 and not solver_failed) or (pso_failed and not solver_failed):
            pso_worse += 1
        else:
            pso_same += 1
        
        if data["time_comparison"][i] > 0 and not solver_failed and not pso_failed:
            solver_faster += 1
        elif data["time_comparison"][i] < 0 and not solver_failed and not pso_failed:
            solver_slower += 1
        
        pso_failed = False
        solver_failed = False

    print(f"Total Simulations: {N}")
    print(f"Failed Simulations: {failed_both} ({(failed_both/N)*100:.2f}%)")
    print(f"Failed Solver Runs: {failed_sol} ({(failed_sol/N)*100:.2f}%)")
    print(f"Failed PSO Runs: {failed_pso} ({(failed_pso/N)*100:.2f}%)")
    print(f"Average Time in PSO: {time_spent_in_pso:.2f} seconds")
    print(f"Average Time in Solver with PSO: {time_solver_Wpso:.2f} seconds")
    print(f"Average Time in Solver without PSO: {time_solver_WOpso:.2f} seconds")
    print(f"Average Total Time: {total_time:.2f} seconds")
    print(f"PSO Better Solutions: {pso_better} ({(pso_better/(N))*100:.2f}%)")
    print(f"PSO Worse Solutions: {pso_worse} ({(pso_worse/(N))*100:.2f}%)")
    print(f"PSO Same Solutions: {pso_same} ({(pso_same/(N))*100:.2f}%)")
    print(f"Solver Faster with PSO: {solver_faster} ({(solver_faster/(N-failed_sol-failed_pso + failed_both))*100:.2f}%)")
    print(f"Solver Slower with PSO: {solver_slower} ({(solver_slower/(N-failed_sol-failed_pso + failed_both))*100:.2f}%)")

    # ------------------------------
    # Prepare Data for Summary Plot
    # ------------------------------

    summary_data = {
        'total_sims': N,
        'failed_both': failed_both,
        'failed_sol': failed_sol,
        'failed_pso': failed_pso,
        'pso_better': pso_better,
        'pso_worse': pso_worse,
        'pso_same': pso_same,
        'solver_faster': solver_faster,
        'solver_slower': solver_slower
    }

    # ------------------------------
    # Generate Plots
    # ------------------------------

    # 2. Summary Bar Graph
    plot_data(summary_data, save_name)
