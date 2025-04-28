import argparse
import csv
import os

def parse_iterations_summary(file_path):
    # Initialize a dictionary to store the extracted values
    results = {}
    results["code"] = "present"

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Check for desired keys and extract their values
            if "Reference version of ComputeSPMV" in line:
                results["code"] = "baseline"
            if "Machine Summary::Distributed Processes" in line:
                results["n_procs"] = int(line.split('=')[-1].strip())
            elif "Benchmark Time Summary::Ortho" in line:
                results["mxp_ortho_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary::SpMV" in line:
                results["mxp_spmv_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary::MG" in line:
                results["mxp_mg_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary:: SpTRSV" in line:
                results["mxp_gs_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary:: Restic" in line:
                results["mxp_restrict_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary:: - Ortho   (reference)" in line:
                results["double_ortho_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary:: - SpMV    (reference)" in line:
                results["double_spmv_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary:: - MG      (reference)" in line:
                results["double_mg_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary:: -  SpTRSV (reference)" in line:
                results["double_gs_time"] = float(line.split('=')[-1].strip())
            elif "Benchmark Time Summary:: -  Restic (reference)" in line:
                results["double_restrict_time"] = float(line.split('=')[-1].strip())

            elif "GFLOP/s Summary::Total for benchmark" in line:
                results['mxp_total_gflops'] = float(line.split('=')[-1].strip())
            elif "GFLOP/s Summary:: - Total     (reference)" in line:
                results['double_total_gflops'] = float(line.split('=')[-1].strip())
            elif "GFLOP/s Summary::Raw Total" in line:
                results['mxp_raw_gflops'] = float(line.split('=')[-1].strip())
            elif "GFLOP/s Summary::Raw Orho" in line:
                results['mxp_ortho_gflops'] = float(line.split('=')[-1].strip())
            elif "GFLOP/s Summary::Raw SpMV" in line:
                results['mxp_spmv_gflops'] = float(line.split('=')[-1].strip())
            elif "GFLOP/s Summary::Raw MG" in line:
                results['mxp_mg_gflops'] = float(line.split('=')[-1].strip())
            elif "GFLOP/s Summary:: - Raw Orho  (reference)" in line:
                results['double_ortho_gflops'] = float(line.split('=')[-1].strip())
            elif "GFLOP/s Summary:: - Raw SpMV  (reference)" in line:
                results['double_spmv_gflops'] = float(line.split('=')[-1].strip())
            elif "GFLOP/s Summary:: - Raw MG    (reference)" in line:
                results['double_mg_gflops'] = float(line.split('=')[-1].strip())

    return results

def save_to_csv(data, output_file):
    # Check if the file exists to determine if we need to write a header
    file_exists = os.path.isfile(output_file)

    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Parse iteration summary from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the input text file')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')
    
    # Parse the command line arguments
    args = parser.parse_args()

    results = parse_iterations_summary(args.file_path)
    
    # Save the results to a CSV file
    save_to_csv(results, args.output_file)
    print(f'Results saved to {args.output_file}')

