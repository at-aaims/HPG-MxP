import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})
#colors = ['r','b','g','c','br']
colors = ['lightgreen', 'skyblue', 'orange', 'lightcoral', 'green', 'blue', 'red', 'purple']

# nprocs per node
npn = 8

def read_csv_and_plot_pie_chart(csv_file, output_prefix):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    #df = df[df["n_procs"] == n_procs]

    if df.empty:
        print(f"No data found.")
        return

    df["n_procs"] = df["n_procs"] // npn
    num_bars = df["n_procs"].nunique()
    
    dist = 4
    bwide = 1.5
    x_base = np.arange(0, dist*num_bars, dist)
    fig,ax = plt.subplots()

    for icode, code_type in enumerate(["double", "mxp"]):
        # Extract timing information
        time_metrics = {
            'SpMV': code_type+'_spmv_time',
            'GS': code_type+'_gs_time',
            'Restr': code_type + "_restrict_time",
            'Orth': code_type+'_ortho_time'
        }

        total_times = df[code_type+"_spmv_time"] + df[code_type+"_gs_time"] \
                + df[code_type+"_restrict_time"] + df[code_type+"_ortho_time"]

        # Filter out any metrics with zero values to avoid issues with the pie chart
        #time_metrics = {k: v for k, v in time_metrics.items() if v > 0}

        # Generate a pie chart
        bot = [0.0,]*num_bars
        x = x_base + icode*bwide
        for icomp,component in enumerate(time_metrics):
            ax.bar(x, df[time_metrics[component]] / total_times, width=bwide, bottom=bot,
                   label=code_type + " " + component, color=colors[icode*4 + icomp])
            bot = df[time_metrics[component]]/total_times + bot
        #plt.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.
        ax.set_xticks(x_base + bwide/2)
        ax.set_xticklabels(df["n_procs"])
        plt.xlabel("Number of nodes")
        plt.ylabel("Time fraction")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.grid(True)
    plt.savefig(output_prefix + "_stacked.png",
            dpi=300, bbox_inches="tight")
        #plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse iteration summary from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the input text file')
    parser.add_argument('output_prefix', type=str, help='Path to the output CSV file')
    
    # Parse the command line arguments
    args = parser.parse_args()
    # Set the path to your CSV file
    read_csv_and_plot_pie_chart(args.file_path, args.output_prefix)
