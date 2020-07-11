//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelParamCalculator.cuda.hh
//---------------------------------------------------------------------------//
#ifndef base_KernelParamCalculator_cuda_hh
#define base_KernelParamCalculator_cuda_hh

#include <cstddef>
#include <cuda_runtime_api.h>
#include "Macros.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Kernel management helper functions.
 *
 * We assume that all our kernel launches use 1-D thread indexing to make
 * things easy. The \c dim_type alias should be the same size as the type of a
 * single \c dim3 member (x/y/z).
 *
 * \code
    KernelParamCalculator calc_kernel_params;
    auto params = calc_kernel_params(states.size());
    my_kernel<<<params.grid_size, params.block_size>>>(kernel_args...);
   \endcode
 */
class KernelParamCalculator
{
  public:
    //@{
    //! Type aliases
    using dim_type = unsigned int;
    //@}

    struct LaunchParams
    {
        dim3 grid_size;  //!< Number of blocks for kernel grid
        dim3 block_size; //!< Number of threads per block
    };

  public:
    // Construct with defaults
    explicit KernelParamCalculator(dim_type block_size = 256u);

    // Get launch parameters
    LaunchParams operator()(size_type min_num_threads) const;

    // Get the thread ID
    inline CELER_FUNCTION static ThreadId thread_id();

  private:
    //! Default threads per block
    dim_type block_size_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "KernelParamCalculator.cuda.i.hh"

#endif // base_KernelParamCalculator_cuh
