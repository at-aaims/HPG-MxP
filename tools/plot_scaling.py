import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

markers = ['o', 's', '^', '>', '<']
lines = ['-', ':', '-.', '--']
plt.rcParams.update({'font.size': 16})

def plot_timings_over_n_procs(csv_file, npn, metrics, output_pref):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    df_cur = df[df["code"] == "present"].reset_index(drop=True)
    df_old = df[df["code"] == "sota"].reset_index(drop=True)

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot each specified metric against n_procs
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in the DataFrame.")
            return
        metlabel = metric.split("_")[0]
        if not df_cur.empty:
            plt.semilogx(df_cur['n_procs']/npn, df_cur[metric]/df_cur["n_procs"], marker=markers[i],
                         ls=lines[i], label=df_cur.at[0,"code"] + " " + metlabel)
        if not df_old.empty:
            plt.semilogx(df_old['n_procs']/npn, df_old[metric]/df_old["n_procs"], marker=markers[i],
                         ls=lines[i], label=df_old.at[0,"code"] + " " + metlabel)
    plt.ylabel('GFLOPS/GCD')

    # Add labels and title
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.xlabel('Number of nodes')
    #plt.title('Timing Metrics vs. Number of Processors')
    plt.grid()
    x_ticks = np.unique((df['n_procs']/npn).astype('int32'))  # Use unique values from the DataFrame
    tixlabels = [val if val != 8192 else '' for val in x_ticks]
    plt.xticks(x_ticks, labels=tixlabels)

    # Save the plot to a file
    plt.savefig(output_pref + ".png", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free up memory

    print(f'Graph saved to {output_pref}.png')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot timing metrics from a CSV file.\
            Assumes a column "code" with values "present" or "sota".')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file')
    parser.add_argument('output_prefix', type=str, help='Path to the output graph file (image)')
    parser.add_argument('--nprocs_per_node', type=int, default=8, help='Number of GPUs per node')

    # Parse the command line arguments
    args = parser.parse_args()

    # Specify the timing metrics you want to plot
    metrics_to_plot = [
        'mxp_total_gflops',
        'double_total_gflops',
        # Add more metrics as needed
    ]

    # Call the plotting function
    plot_timings_over_n_procs(args.input_csv, args.nprocs_per_node, metrics_to_plot,
                              args.output_prefix)
