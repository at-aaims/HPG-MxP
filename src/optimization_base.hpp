#ifndef HPGMP_OPTIMIZATION_BASE_HPP
#define HPGMP_OPTIMIZATION_BASE_HPP

/** @brief Base type for implementations to derive custom optimization structures from.
 *
 * The purpose it to use this base type in the SparseMatrix class,
 * replacing the former void pointer.
 */
class OptimizationData
{
public:
    /// Ensures custom derived optimization structures are destroyed correctly.
    virtual ~OptimizationData()
    { }
};

#endif
