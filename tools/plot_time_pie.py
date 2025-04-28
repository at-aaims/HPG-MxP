import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

# nprocs per node
npn = 8

def read_csv_and_plot_pie_chart(csv_file, n_procs, output_prefix):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    df = df[df["n_procs"] == n_procs]

    if df.empty:
        print(f"No data found for n_procs = {n_procs}.")
        return

    # Select specific columns related to time metrics
    # Here we are assuming the timing metrics of interest are formatted in a specific way
    # You may need to adjust the column names in accordance with your CSV
    #print(df["ref_spmv_time"].item())

    for code_type in ["double", "mxp"]:
        # Extract timing information
        time_metrics = {
            'SpMV': df[code_type+'_spmv_time'].item(),
            'GS': df[code_type+'_gs_time'].item(),
            'Restr': df[code_type + "_restrict_time"].item(),
            'Orth': df[code_type+'_ortho_time'].item(),
        }

        # Filter out any metrics with zero values to avoid issues with the pie chart
        time_metrics = {k: v for k, v in time_metrics.items() if v > 0}

        # Generate a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(time_metrics.values(), labels=time_metrics.keys(), autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.
        plt.savefig(output_prefix + "_"+code_type+"_N"+ str(n_procs//npn)+ ".png", dpi=300,
                    bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse iteration summary from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the input text file')
    parser.add_argument('output_prefix', type=str, help='Path to the output CSV file')
    parser.add_argument('--nprocs', type=int, help='Proc count to plot')
    
    # Parse the command line arguments
    args = parser.parse_args()
    # Set the path to your CSV file
    read_csv_and_plot_pie_chart(args.file_path, args.nprocs, args.output_prefix)
