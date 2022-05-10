//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Copier.hh"
#include "corecel/math/Algorithms.hh"
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
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 *
 * This creates the maximum possible number of track initializers on the
 * templated memory space from host primaries (either the number of host
 * primaries that have not yet been initialized or the size of the available
 * storage in the track initializer vector, whichever is smaller).
 */
template<MemSpace M>
inline void extend_from_primaries(const TrackInitParamsHostRef& params,
                                  TrackInitStateData<Ownership::value, M>* data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(data && *data);

    // Number of primaries to initialize
    auto count = min(data->initializers.capacity() - data->initializers.size(),
                     data->num_primaries);
    if (count > 0)
    {
        data->initializers.resize(data->initializers.size() + count);

        // Allocate memory and copy primaries
        Collection<Primary, Ownership::value, M> primaries;
        make_builder(&primaries).resize(count);
        Copier<Primary, MemSpace::host> copy{
            params.primaries[ItemRange<Primary>(
                ItemId<Primary>(data->num_primaries - count),
                ItemId<Primary>(data->num_primaries))]};
        copy(M, primaries[AllItems<Primary, M>{}]);
        data->num_primaries -= count;

        // Create track initializers from primaries
        generated::process_primaries(primaries[AllItems<Primary, M>{}],
                                     make_ref(*data));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Initialize track states
 *
 * Tracks created from secondaries produced in this step will have the geometry
 * state copied over from the parent instead of initialized from the position.
 * If there are more empty slots than new secondaries, they will be filled by
 * any track initializers remaining from previous steps using the position.
 */
template<MemSpace M>
inline void initialize_tracks(const CoreRef<M>& core_data,
                              TrackInitStateData<Ownership::value, M>* data)
{
    CELER_EXPECT(core_data);
    CELER_EXPECT(data && *data);

    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers
    size_type num_tracks
        = std::min(data->vacancies.size(), data->initializers.size());
    if (num_tracks > 0)
    {
        // Launch a kernel to initialize tracks on device
        auto num_vacancies
            = min(data->vacancies.size(), data->initializers.size());
        generated::init_tracks(core_data, make_ref(*data), num_vacancies);
        data->initializers.resize(data->initializers.size() - num_tracks);
        data->vacancies.resize(data->vacancies.size() - num_tracks);
    }
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
extend_from_secondaries(const CoreRef<M>&                        core_data,
                        TrackInitStateData<Ownership::value, M>* data)
{
    CELER_EXPECT(core_data);
    CELER_EXPECT(data && *data);

    // Resize the vector of vacancies to be equal to the number of tracks
    data->vacancies.resize(core_data.states.size());

    // Launch a kernel to identify which track slots are still alive and count
    // the number of surviving secondaries per track
    generated::locate_alive(core_data, make_ref(*data));

    // Remove all elements in the vacancy vector that were flagged as active
    // tracks, leaving the (sorted) indices of the empty slots
    size_type num_vac = detail::remove_if_alive<M>(data->vacancies.data());
    data->vacancies.resize(num_vac);

    // The exclusive prefix sum of the number of secondaries produced by each
    // track is used to get the start index in the vector of track initializers
    // for each thread. Starting at that index, each thread creates track
    // initializers from all surviving secondaries produced in its
    // interaction.
    data->num_secondaries = detail::exclusive_scan_counts<M>(
        data->secondary_counts[AllItems<size_type, M>{}]);

    // TODO: if we don't have space for all the secondaries, we will need to
    // buffer the current track initializers to create room
    CELER_VALIDATE(
        data->num_secondaries + data->initializers.size()
            <= data->initializers.capacity(),
        << "insufficient capacity (" << data->initializers.capacity()
        << ") for track initializers (created " << data->num_secondaries
        << " new secondaries for a total capacity requirement of "
        << data->num_secondaries + data->initializers.size() << ")");

    // Launch a kernel to create track initializers from secondaries
    data->initializers.resize(data->initializers.size()
                              + data->num_secondaries);
    generated::process_secondaries(core_data, make_ref(*data));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
