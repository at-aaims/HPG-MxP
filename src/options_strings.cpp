#include "hpgmp.hpp"

std::string get_string(const validation_t val_type) {
    if(val_type == validation_t::standard) {
        return "standard";
    } else {
        return "fullscale";
    }
}

std::string get_string(const run_t run_type) {
    if(run_type == run_t::benchmark) {
        return "benchmark";
    } else if(run_type == run_t::benchmark_no_ref) {
        return "benchmark_no_ref";
    } else if(run_type == run_t::standalone_ref) {
        return "standalone_ref";
    } else {
        return "standalone_mxp";
    }
}
