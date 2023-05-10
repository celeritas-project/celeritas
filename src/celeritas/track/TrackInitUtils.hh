//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitUtils.hh
//! \brief Helper functions for initializing tracks
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "TrackInitData.hh"
#include "detail/TrackInitAlgorithms.hh"
#include "generated/InitTracks.hh"
#include "generated/LocateAlive.hh"
#include "generated/ProcessPrimaries.hh"
#include "generated/ProcessSecondaries.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
// TODO: move most of these to "detail" namespace
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from a vector of host primary particles.
 */
template<MemSpace M>
inline void extend_from_primaries(CoreParams const& core_params,
                                  CoreState<M>& core_state,
                                  Span<Primary const> host_primaries)
{
    CELER_EXPECT(!host_primaries.empty());

    auto& init = core_state.ref().init;
    CELER_ASSERT(host_primaries.size() + init.scalars.num_initializers
                 <= init.initializers.size());

    // Resizing the initializers is a non-const operation, but the only one.
    init.scalars.num_initializers += host_primaries.size();

    // Allocate memory and copy primaries
    Collection<Primary, Ownership::value, M> primaries;
    resize(&primaries, host_primaries.size());
    Copier<Primary, M> copy_to_temp{primaries[AllItems<Primary, M>{}]};
    copy_to_temp(MemSpace::host, host_primaries);

    // Create track initializers from primaries
    generated::process_primaries(core_params.ref<M>(),
                                 core_state.ref(),
                                 primaries[AllItems<Primary, M>{}]);
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
inline void
initialize_tracks(CoreParams const& core_params, CoreState<M>& core_state)
{
    auto& scalars = core_state.ref().init.scalars;

    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers
    size_type num_tracks
        = std::min(scalars.num_vacancies, scalars.num_initializers);
    if (num_tracks > 0)
    {
        // Launch a kernel to initialize tracks on device
        auto num_vacancies
            = min(scalars.num_vacancies, scalars.num_initializers);
        generated::init_tracks(
            core_params.ref<M>(), core_state.ref(), num_vacancies);
        // Resizing initializers/vacancies is a non-const operation
        scalars.num_initializers = scalars.num_initializers - num_tracks;
        scalars.num_vacancies = scalars.num_vacancies - num_tracks;
    }

    // Store number of active tracks at the start of the loop
    scalars.num_active = core_state.size() - scalars.num_vacancies;
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from secondary particles.
 *
 * Secondaries produced by each track are ordered arbitrarily in memory, and
 * the memory may be fragmented if not all secondaries survived cutoffs. For
 * example, after the interactions have been processed and cutoffs applied, the
 * track states and their secondaries might look like the following (where 'X'
 * indicates a track or secondary that did not survive):
 * \verbatim

   thread ID   | 0   1 2           3       4   5 6           7       8   9
   track ID    | 10  X 8           7       X   5 4           X       2   1
   secondaries | [X]   [X, 11, 12] [13, X] [X]   [14, X, 15] [X, 16] [X]

   \endverbatim
 *
 * Because the order in which threads receive a chunk of memory from the
 * secondary allocator is nondeterministic, the actual ordering of the
 * secondaries in memory is unpredictable; for instance:
 * \verbatim

  secondary storage | [X, 13, X, X, 11, 12, X, X, 16, 14, X, 15, X]

  \endverbatim
 *
 * When track initializers are created from secondaries, they are ordered by
 * thread ID to ensure reproducibility. If a track that produced secondaries
 * has died (e.g., thread ID 7 in the example above), one of its secondaries is
 * immediately used to fill that track slot:
 * \verbatim

   thread ID   | 0   1 2           3       4   5 6           7       8   9
   track ID    | 10  X 8           7       X   5 4           16      2   1
   secondaries | [X]   [X, 11, 12] [13, X] [X]   [14, X, 15] [X, X]  [X]

   \endverbatim
 *
 * This way, the geometry state is reused rather than initialized from the
 * position (which is expensive). This also prevents the geometry state from
 * being overwritten by another track's secondary, so if the track produced
 * multiple secondaries, the rest are still able to copy the parent's state.
 *
 * Track initializers are created from the remaining secondaries and are added
 * to the back of the vector. The thread ID of each secondary's parent is also
 * stored, so any new tracks initialized from secondaries produced in this
 * step can copy the geometry state from the parent. The indices of the empty
 * slots in the track vector are identified and stored as a sorted vector of
 * vacancies.
 * \verbatim

   track initializers | 11 12 13 14 15
   parent             | 2  2  3  6  6
   vacancies          | 1  4

   \endverbatim
 */
template<MemSpace M>
inline void
extend_from_secondaries(CoreParams const& core_params, CoreState<M>& core_state)
{
    TrackInitStateData<Ownership::reference, M>& init = core_state.ref().init;

    // Launch a kernel to identify which track slots are still alive and count
    // the number of surviving secondaries per track
    generated::locate_alive(core_params.ref<M>(), core_state.ref());

    // Remove all elements in the vacancy vector that were flagged as active
    // tracks, leaving the (sorted) indices of the empty slots
    init.scalars.num_vacancies = detail::remove_if_alive(init.vacancies);

    // The exclusive prefix sum of the number of secondaries produced by each
    // track is used to get the start index in the vector of track initializers
    // for each thread. Starting at that index, each thread creates track
    // initializers from all surviving secondaries produced in its
    // interaction.
    init.scalars.num_secondaries
        = detail::exclusive_scan_counts(init.secondary_counts);

    // TODO: if we don't have space for all the secondaries, we will need to
    // buffer the current track initializers to create room
    init.scalars.num_initializers += init.scalars.num_secondaries;
    CELER_VALIDATE(init.scalars.num_initializers <= init.initializers.size(),
                   << "insufficient capacity (" << init.initializers.size()
                   << ") for track initializers (created "
                   << init.scalars.num_secondaries
                   << " new secondaries for a total capacity requirement of "
                   << init.scalars.num_initializers << ")");

    // Launch a kernel to create track initializers from secondaries
    init.scalars.num_alive = core_state.size() - init.scalars.num_vacancies;
    generated::process_secondaries(core_params.ref<M>(), core_state.ref());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
