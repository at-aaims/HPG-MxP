#ifndef HPGMP_PERF_COUNTER_HPP
#define HPGMP_PERF_COUNTER_HPP

#include "DataTypes.hpp"

/** Stores the number of floating point operations (flops) and
 * loads and stores from memory (mem_traffic) in any kernel.
 *
 * Stores flops and memory traffic separately for every precision used,
 * stored highest precision to lowest precision.
 */
class flops_and_traffic {
public: 
    static constexpr int n_precs = 2;

    /// Get index in counter arrays based on precision type. 
    template <typename T> 
    constexpr std::enable_if_t<std::is_floating_point<T>::value, int> index() const
    {
        if constexpr (std::is_same<T, double>::value) {
            return 0;
        } else if constexpr (std::is_same<T,float>::value) {
            return 1;
        } else if constexpr (std::is_same<T,half>::value) {
            return 2;
        } else {
            static_assert(false, "Precision not supported!");
        }
    }

    template <typename T> 
    std::enable_if_t<std::is_floating_point<T>::value> add_flops(const double count)
    {
        constexpr int idx = index<T>();
        flops[idx] += count;
    }

    template <typename T> 
    std::enable_if_t<std::is_floating_point<T>::value, double> get_flops() const
    {
        constexpr int idx = index<T>();
        return flops[idx];
    }

    double get_total_flops() const
    {
        double total = 0;
        for (auto x : flops) {
            total += x;
        }
        return total;
    }

    template <typename T> 
    std::enable_if_t<std::is_floating_point<T>::value> add_memory_traffic(const double count)
    {
        constexpr int idx = index<T>();
        f_mem_traffic[idx] += count;
    }

    template <typename T>
    std::enable_if_t<std::is_same<T,int>::value> add_memory_traffic(const double count)
    {
        i_mem_traffic += count;
    }

    template <typename T> 
    std::enable_if_t<std::is_floating_point<T>::value, double> get_memory_traffic() const
    {
        constexpr int idx = index<T>();
        return f_mem_traffic[idx];
    }

    template <typename T>
    std::enable_if_t<std::is_same<T,int>::value, double> get_memory_traffic() const
    {
        return i_mem_traffic;
    }

    double get_total_memory_bytes() const
    {
        double total = i_mem_traffic * sizeof(int);
        constexpr std::array<size_t,3> sizes{sizeof(double), sizeof(float), sizeof(half)};
        for(int i = 0; i < n_precs; i++) {
            total += static_cast<double>(f_mem_traffic[i])*sizes[i];
        }
        return total;
    }

    void reset() {
        flops = {};
        f_mem_traffic = {};
        i_mem_traffic = {};
        //i_mem_traffic = 0.0;
        //for(int i = 0; i < n_precs; i++) {
        //    flops[i] =0.0;
        //    f_mem_traffic[i] = 0.0;
        //}
    }

private:
    std::array<double, n_precs> flops = {};
    std::array<double, n_precs> f_mem_traffic = {};
    double i_mem_traffic{};
};

/** Floating-point operations and memory traffic counters for different parts of HPGMP.
 */
struct perf_counters {
    flops_and_traffic mg_gs;    ///< Gauss-Seidel
    flops_and_traffic mg_rp;    ///< Restriction (-SpMV fused) and prolongation
    flops_and_traffic ortho;    ///< CGS2 orthogonalization
    flops_and_traffic spmv;     ///< SpMV outside MG
    flops_and_traffic vecupd;   ///< Vector update operations WAXPBY and scale
    flops_and_traffic qr_host;  ///< QR factorization update on host

    /// Reset all counters to zero
    void reset() {
        mg_gs.reset();
        mg_rp.reset();
        ortho.reset();
        spmv.reset();
        vecupd.reset();
        qr_host.reset();
    }
};

#endif
