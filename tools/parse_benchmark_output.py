import argparse
import csv
import os

def get_membw_line(results, line, part):
    """
    @param results  Output dict to add keys to.
    @param line  Inputs file line to parse.
    @param part  Prefix string for output key.
    """
    if "Raw Total" in line:
        results[part+'_raw'] = float(line.split('=')[-1].strip())
    elif "Raw Ortho" in line:
        results[part+'_ortho'] = float(line.split('=')[-1].strip())
    elif "Raw SpMV" in line:
        results[part+'_spmv'] = float(line.split('=')[-1].strip())
    elif "Raw MG" in line:
        results[part+'_mg'] = float(line.split('=')[-1].strip())
    elif "GS" in line:
        results[part+'_mg_gs'] = float(line.split('=')[-1].strip())
    elif "RestrPro" in line:
        results[part+'_mg_rp'] = float(line.split('=')[-1].strip())
    elif "Raw vecupd" in line:
        results[part+'_vecupd'] = float(line.split('=')[-1].strip())

def parse_iterations_summary(file_path, extr_type):
    """ Parse the HPG-MxP output file for the needed metrics.

    @param file_path  Path to HPG-MxP output file.
    @param extr_type  Type of metrics to extract: "flops" or "membw"
    """
    results = {}
    results["code"] = "present"

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            if "Reference version of ComputeSPMV" in line:
                results["code"] = "baseline"
            elif "Machine Summary::Distributed Processes" in line:
                results["n_procs"] = int(line.split('=')[-1].strip())

            if extr_type == "flops":
                if "Benchmark Time Summary::Ortho" in line:
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
            elif extr_type == "membw":
                if "Memory Bandwidth (GB/s)" in line:
                    if "mxp" in line:
                        get_membw_line(results, line, "mxp")
                    elif "Total for benchmark" in line:
                        results['mxp_total'] = float(line.split('=')[-1].strip())
                    elif "reference" in line:
                        get_membw_line(results, line, "double")
            else:
                raise "Invalid extraction type!"

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
    parser.add_argument('-e', '--extraction_type', type=str, help='Type of data to extract - "flops", "membw"')
    
    # Parse the command line arguments
    args = parser.parse_args()

    results = parse_iterations_summary(args.file_path, args.extraction_type)
    
    # Save the results to a CSV file
    save_to_csv(results, args.output_file)
    print(f'Results saved to {args.output_file}')

