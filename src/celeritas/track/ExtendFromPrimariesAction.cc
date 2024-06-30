//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromPrimariesAction.cc
//---------------------------------------------------------------------------//
#include "ExtendFromPrimariesAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/ProcessPrimariesExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void ExtendFromPrimariesAction::execute(CoreParams const& params,
                                        CoreStateHost& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void ExtendFromPrimariesAction::execute(CoreParams const& params,
                                        CoreStateDevice& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Construct primaries.
 */
template<MemSpace M>
void ExtendFromPrimariesAction::execute_impl(CoreParams const& params,
                                             CoreState<M>& state) const
{
    auto primary_range = state.primary_range();
    if (primary_range.empty() && !state.warming_up())
        return;

    auto primaries = state.primary_storage()[primary_range];

    // Create track initializers from primaries
    state.counters().num_initializers += primaries.size();
    this->process_primaries(params, state, primaries);

    // Mark that the primaries have been processed
    state.clear_primaries();
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to create track initializers from primary particles.
 *
 * This function loops over *primaries and initializers*, not *track slots*, so
 * we do not use \c launch_core or \c launch_action .
 */
void ExtendFromPrimariesAction::process_primaries(
    CoreParams const&, CoreStateHost& state, Span<Primary const> primaries) const
{
    MultiExceptionHandler capture_exception;
    detail::ProcessPrimariesExecutor execute_thread{
        state.ptr(), primaries, state.counters()};
#if defined(_OPENMP) && CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp parallel for
#endif
    for (size_type i = 0, size = primaries.size(); i != size; ++i)
    {
        CELER_TRY_HANDLE(execute_thread(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ExtendFromPrimariesAction::process_primaries(CoreParams const&,
                                                  CoreStateDevice&,
                                                  Span<Primary const>) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
