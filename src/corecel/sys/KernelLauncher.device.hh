//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelLauncher.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"

#include "Device.hh"
#include "KernelParamCalculator.device.hh"
#include "KernelTraits.hh"
#include "Stream.hh"
#include "ThreadId.hh"

#include "detail/KernelLauncherImpl.device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Profile and launch Celeritas kernels.
 *
 * The template argument \c F may define a member type named \c Applier.
 * \c F::Applier should have up to two static constexpr int variables named
 * \c max_block_size and/or \c min_warps_per_eu.
 * If present, the kernel will use appropriate \c __launch_bounds__.
 * If \c F::Applier::min_warps_per_eu exists then \c F::Applier::max_block_size
 * must also be present or we get a compile error.
 *
 * The semantics of the second \c __launch_bounds__ argument differs between
 * CUDA and HIP.  \c KernelLauncher expects HIP semantics. If Celeritas is
 * built targeting CUDA, it will automatically convert that argument to match
 * CUDA semantics.
 *
 * The CUDA-specific 3rd argument \c maxBlocksPerCluster is not supported.
 *
 * Example:
 * \code
 void FooAction::launch_kernel(size_type count) const
 {
    auto execute_thread = make_blah_executor(blah);
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(*this);
    launch_kernel(state, execute_thread);
 }
 * \endcode
 */
template<class F>
class KernelLauncher
{
    static_assert((std::is_trivially_copyable_v<F> || CELERITAS_USE_HIP
                   || CELER_COMPILER == CELER_COMPILER_CLANG)
                      && !std::is_pointer_v<F> && !std::is_reference_v<F>,
                  "Launched action must be a trivially copyable function "
                  "object");

  public:
    //! Create a launcher from a label
    explicit KernelLauncher(std::string_view name)
        : calc_launch_params_{name, &detail::launch_action_impl<F>}
    {
    }

    //! Launch a kernel for a thread range
    void operator()(Range<ThreadId> threads,
                    StreamId stream_id,
                    F const& call_thread) const
    {
        if (!threads.empty())
        {
            using StreamT = CELER_DEVICE_PREFIX(Stream_t);
            StreamT stream = celeritas::device().stream(stream_id).get();
            auto config = calc_launch_params_(threads.size());
            detail::launch_action_impl<F>
                <<<config.blocks_per_grid, config.threads_per_block, 0, stream>>>(
                    threads, call_thread);
        }
    }

    //! Launch a kernel with a custom number of threads
    void operator()(size_type num_threads,
                    StreamId stream_id,
                    F const& call_thread) const
    {
        CELER_EXPECT(num_threads > 0);
        CELER_EXPECT(stream_id);
        (*this)(range(ThreadId{num_threads}), stream_id, call_thread);
    }

  private:
    KernelParamCalculator calc_launch_params_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
