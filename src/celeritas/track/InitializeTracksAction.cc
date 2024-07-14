//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.cc
//---------------------------------------------------------------------------//
#include "InitializeTracksAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "detail/InitTracksExecutor.hh"  // IWYU pragma: associated
#include "detail/TrackInitAlgorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void InitializeTracksAction::execute(CoreParams const& params,
                                     CoreStateHost& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void InitializeTracksAction::execute(CoreParams const& params,
                                     CoreStateDevice& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize track states.
 *
 * Tracks created from secondaries produced in this step will have the geometry
 * state copied over from the parent instead of initialized from the position.
 * If there are more empty slots than new secondaries, they will be filled by
 * any track initializers remaining from previous steps using the position.
 */
template<MemSpace M>
void InitializeTracksAction::execute_impl(CoreParams const& core_params,
                                          CoreState<M>& core_state) const
{
    auto& counters = core_state.counters();

    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers
    size_type num_new_tracks
        = std::min(counters.num_vacancies, counters.num_initializers);
    if (num_new_tracks > 0 || core_state.warming_up())
    {
        size_type partition_index{};
        if (core_params.init()->host_ref().track_order
            == TrackOrder::partition_data)
        {
            // Partition tracks by whether they are charged or neutral
            partition_index = detail::partition_initializers(
                core_params,
                core_state.ref().init.initializers,
                counters,
                num_new_tracks,
                core_state.stream_id());
        }

        // Launch a kernel to initialize tracks
        this->execute_impl(
            core_params, core_state, num_new_tracks, partition_index);

        // Update initializers/vacancies
        counters.num_initializers -= num_new_tracks;
        counters.num_vacancies -= num_new_tracks;
    }

    // Store number of active tracks at the start of the loop
    counters.num_active = core_state.size() - counters.num_vacancies;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to initialize tracks.
 *
 * The thread index here corresponds to initializer indices, not track slots
 * (or indicies into the track slot indirection array).
 */
void InitializeTracksAction::execute_impl(CoreParams const& core_params,
                                          CoreStateHost& core_state,
                                          size_type num_new_tracks,
                                          size_type partition_index) const
{
    MultiExceptionHandler capture_exception;
    detail::InitTracksExecutor execute_thread{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        num_new_tracks,
        partition_index,
        core_state.counters()};
#if defined(_OPENMP) && CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp parallel for
#endif
    for (size_type i = 0; i != num_new_tracks; ++i)
    {
        CELER_TRY_HANDLE(execute_thread(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void InitializeTracksAction::execute_impl(CoreParams const&,
                                          CoreStateDevice&,
                                          size_type,
                                          size_type) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
