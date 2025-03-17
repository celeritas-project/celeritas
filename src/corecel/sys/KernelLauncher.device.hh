//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelLauncher.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>
#include <type_traits>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"

#include "Device.hh"
#include "KernelParamCalculator.device.hh"
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
 void launch_kernel(DeviceParams const& params, size_type count) const
 {
    auto execute_thread = BlahExecutor{params};
    static KernelLauncher<decltype(execute_thread)> const launch("blah");
    launch_kernel(count, StreamId{}, execute_thread);
 }
 * \endcode
 */
template<class F>
class KernelLauncher
{
    static_assert(
        (std::is_trivially_copyable_v<F> || CELERITAS_USE_HIP
         || CELER_COMPILER == CELER_COMPILER_CLANG)
            && !std::is_pointer_v<F> && !std::is_reference_v<F>,
        R"(Launched action must be a trivially copyable function object)");

  public:
    // Create a launcher from a label
    explicit inline KernelLauncher(std::string_view name);

    // Launch a kernel for a thread range
    inline void operator()(Range<ThreadId> threads,
                           StreamId stream_id,
                           F const& execute_thread) const;

    // Launch a kernel with a custom number of threads
    inline void operator()(size_type num_threads,
                           StreamId stream_id,
                           F const& execute_thread) const;

  private:
    KernelParamCalculator calc_launch_params_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Create a launcher from a label.
 */
template<class F>
KernelLauncher<F>::KernelLauncher(std::string_view name)
    : calc_launch_params_{name, &detail::launch_action_impl<F>}
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel for a thread range.
 */
template<class F>
void KernelLauncher<F>::operator()(Range<ThreadId> threads,
                                   StreamId stream_id,
                                   F const& execute_thread) const
{
    if (!threads.empty())
    {
        using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
        StreamT stream = stream_id
                             ? celeritas::device().stream(stream_id).get()
                             : nullptr;
        auto config = calc_launch_params_(threads.size());
        detail::launch_action_impl<F>
            <<<config.blocks_per_grid, config.threads_per_block, 0, stream>>>(
                threads, execute_thread);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel with a custom number of threads.
 *
 * The launch arguments have the same ordering as CUDA/HIP kernel launch
 * arguments.
 *
 * \param num_threads Total number of active consecutive threads
 * \param stream_id Execute the kernel on this device stream
 * \param execute_thread Call the given functor with the thread ID
 */
template<class F>
void KernelLauncher<F>::operator()(size_type num_threads,
                                   StreamId stream_id,
                                   F const& execute_thread) const
{
    (*this)(range(ThreadId{num_threads}), stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
