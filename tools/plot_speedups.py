import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

markers = ['o', 's', '^', '>', '<']
lines = ['--', ':', '-.', '-']
plt.rcParams.update({'font.size': 14})

def plot_timings_over_n_procs(csv_file, npn, metrics, output_pref):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Print the DataFrame (optional, for debugging)
    #print("Data from CSV:\n", df)

    df_cur = df[df["code"] == "present"].reset_index(drop=True)
    df_old = df[df["code"] == "sota"].reset_index(drop=True)

    # Set up the plot
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(9, 3))

    nmetrics = len(metrics)

    # Plot each specified metric against n_procs
    for i, metric in enumerate(metrics):
        if "double_"+metric not in df.columns or "mxp_" + metric not in df.columns:
            print(f"Warning: Metric opt/ref '{metric}' not found in the DataFrame.")
            return
        metlabel = metric.split("_")[0]
        if not df_cur.empty:
            if nmetrics == 1:
                lss = lines[-1]
                marks = markers[-1]
            else:
                lss = lines[i]
                marks = markers[i]
            plt.semilogx(df_cur['n_procs']/npn, df_cur["mxp_"+metric]/df_cur["double_"+metric],
                         ls=lss, marker=marks, label=df_cur.at[0,"code"] + " " + metlabel)
        if not df_old.empty:
            plt.semilogx(df_old['n_procs']/npn, df_old["mxp_"+metric]/df_old["double_"+metric],
                         ls=lines[i], marker=markers[i], label=df_old.at[0,"code"] + " " + metlabel)
    plt.ylabel("MxP speedup")
    #plt.ylim(bottom=1.2, top=1.8)
    plt.ylim(bottom=0.98)

    # Add labels and title
    if nmetrics == 1:
        plt.legend(loc="best")
    else:
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.xlabel('Number of nodes')
    plt.grid()

    x_ticks = np.unique((df['n_procs']/npn).astype('int32'))
    # Remove 8192; change this line
    tixlabels = [val if val != 8192 else '' for val in x_ticks]
    plt.xticks(x_ticks, labels=tixlabels)

    # Save the plot to a file
    plt.savefig(output_pref + ".png", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free up memory

    print(f'Graph saved to {output_pref}.png')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot timing metrics from a CSV file.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file')
    parser.add_argument('output_prefix', type=str, help='Path to the output graph file (image)')
    parser.add_argument('--nprocs_per_node', type=int, default=8, help='Number of GPUs per node')

    # Parse the command line arguments
    args = parser.parse_args()

    # Specify the timing metrics you want to plot
    print("Plotting mxp speedups")
    metrics_to_plot = ['ortho_gflops', 'spmv_gflops', 'mg_gflops']

    # Call the plotting function
    plot_timings_over_n_procs(args.input_csv, args.nprocs_per_node, metrics_to_plot,
                              args.output_prefix+"split")
    
    metrics_to_plot = ['total_gflops']

    # Call the plotting function
    plot_timings_over_n_procs(args.input_csv, args.nprocs_per_node, metrics_to_plot,
                              args.output_prefix+"total")
