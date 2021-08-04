//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitUtils.cc
//---------------------------------------------------------------------------//
#include "TrackInitUtils.hh"

#include "base/Algorithms.hh"
#include "detail/InitializeTracks.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from primary particles.
 *
 * This creates the maximum possible number of track initializers on device
 * from host primaries (either the number of host primaries that have not yet
 * been initialized on device or the size of the available storage in the track
 * initializer vector, whichever is smaller).
 */
void extend_from_primaries(const TrackInitParamsHostRef& params,
                           TrackInitStateDeviceVal*      data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(data && *data);

    // Number of primaries to copy to device
    auto count = min(data->initializers.capacity() - data->initializers.size(),
                     data->num_primaries);
    if (count)
    {
        data->initializers.resize(data->initializers.size() + count);

        // Allocate memory on device and copy primaries
        DeviceVector<Primary> primaries(count);
        primaries.copy_to_device(params.primaries[ItemRange<Primary>(
            ItemId<Primary>(data->num_primaries - count),
            ItemId<Primary>(data->num_primaries))]);
        data->num_primaries -= count;

        // Launch a kernel to create track initializers from primaries
        detail::process_primaries(primaries.device_pointers(), make_ref(*data));
    }
}

void extend_from_primaries(const TrackInitParamsHostRef& params,
                           TrackInitStateHostVal*        data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(data && *data);

    // Number of primaries to copy to device
    auto count = min(data->initializers.capacity() - data->initializers.size(),
                     data->num_primaries);
    if (count)
    {
        data->initializers.resize(data->initializers.size() + count);

        // Allocate memory on device and copy primaries
        auto primaries = params.primaries[ItemRange<Primary>(
            ItemId<Primary>(data->num_primaries - count),
            ItemId<Primary>(data->num_primaries))];
        data->num_primaries -= count;

        // Launch a kernel to create track initializers from primaries
        detail::process_primaries(primaries, make_ref(*data));
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
void extend_from_secondaries(const ParamsDeviceRef&   params,
                             const StateDeviceRef&    states,
                             TrackInitStateDeviceVal* data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(states);
    CELER_EXPECT(data && *data);

    // Resize the vector of vacancies to be equal to the number of tracks
    data->vacancies.resize(states.size());

    // Launch a kernel to identify which track slots are still alive and count
    // the number of surviving secondaries per track
    detail::locate_alive(params, states, make_ref(*data));

    // Remove all elements in the vacancy vector that were flagged as active
    // tracks, leaving the (sorted) indices of the empty slots
    size_type num_vac = detail::remove_if_alive(data->vacancies.pointers());
    data->vacancies.resize(num_vac);

    // Sum the total number secondaries produced in all interactions
    // TODO: if we don't have space for all the secondaries, we will need to
    // buffer the current track initializers to create room
    size_type num_secondaries = detail::reduce_counts(
        data->secondary_counts[AllItems<size_type, MemSpace::device>{}]);
    CELER_VALIDATE(num_secondaries + data->initializers.size()
                       <= data->initializers.capacity(),
                   << "insufficient capacity (" << data->initializers.capacity()
                   << ") for track initializers (created " << num_secondaries
                   << " new secondaries for a total capacity requirement of "
                   << num_secondaries + data->initializers.size() << ")");

    // The exclusive prefix sum of the number of secondaries produced by each
    // track is used to get the start index in the vector of track initializers
    // for each thread. Starting at that index, each thread creates track
    // initializers from all surviving secondaries produced in its
    // interaction.
    detail::exclusive_scan_counts(
        data->secondary_counts[AllItems<size_type, MemSpace::device>{}]);

    // Launch a kernel to create track initializers from secondaries
    data->parents.resize(num_secondaries);
    data->initializers.resize(data->initializers.size() + num_secondaries);
    detail::process_secondaries(params, states, make_ref(*data));
}

//---------------------------------------------------------------------------//
/*!
 * Initialize track states on device.
 *
 * Tracks created from secondaries produced in this step will have the geometry
 * state copied over from the parent instead of initialized from the position.
 * If there are more empty slots than new secondaries, they will be filled by
 * any track initializers remaining from previous steps using the position.
 */
void initialize_tracks(const ParamsDeviceRef&   params,
                       const StateDeviceRef&    states,
                       TrackInitStateDeviceVal* data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(states);
    CELER_EXPECT(data && *data);

    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers
    size_type num_tracks
        = std::min(data->vacancies.size(), data->initializers.size());
    if (num_tracks > 0)
    {
        // Launch a kernel to initialize tracks on device
        detail::init_tracks(params, states, make_ref(*data));
        data->initializers.resize(data->initializers.size() - num_tracks);
        data->vacancies.resize(data->vacancies.size() - num_tracks);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
