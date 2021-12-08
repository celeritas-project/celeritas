//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.cc
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include <numeric>
#include "base/Atomics.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "sim/SimTrackView.hh"
#include "sim/TrackInitData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states on host.
 */
void init_tracks(const ParamsHostRef&         params,
                 const StateHostRef&          states,
                 const TrackInitStateHostRef& data)
{
    // Number of vacancies, limited by the initializer size
    auto num_vacancies
        = std::min(data.vacancies.size(), data.initializers.size());

    InitTracksLauncher<MemSpace::host> launch(params, states, data);
    for (auto tid : range(ThreadId{num_vacancies}))
    {
        launch(tid);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find empty slots in the track vector and count secondaries created.
 */
void locate_alive(const ParamsHostRef&         params,
                  const StateHostRef&          states,
                  const TrackInitStateHostRef& data)
{
    LocateAliveLauncher<MemSpace::host> launch(params, states, data);
    for (auto tid : range(ThreadId{states.size()}))
    {
        launch(tid);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from primary particles.
 */
void process_primaries(Span<const Primary>          primaries,
                       const TrackInitStateHostRef& data)
{
    // TODO: What to do about celeritas::size_type for host?
    ProcessPrimariesLauncher<MemSpace::host> launch(primaries, data);
    for (auto tid :
         range(ThreadId{static_cast<celeritas::size_type>(primaries.size())}))
    {
        launch(tid);
    }
}
//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from secondary particles.
 */
void process_secondaries(const ParamsHostRef&         params,
                         const StateHostRef&          states,
                         const TrackInitStateHostRef& data)
{
    ProcessSecondariesLauncher<MemSpace::host> launch(params, states, data);
    for (auto tid : range(ThreadId{states.size()}))
    {
        launch(tid);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Remove all elements in the vacancy vector that were flagged as active
 * tracks.
 */
template<>
size_type remove_if_alive<MemSpace::host>(Span<size_type> vacancies)
{
    auto end = std::remove_if(vacancies.data(),
                              vacancies.data() + vacancies.size(),
                              IsEqual{flag_id()});

    // New size of the vacancy vector
    size_type result = end - vacancies.data();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sum the total number of surviving secondaries.
 */
template<>
size_type reduce_counts<MemSpace::host>(Span<size_type> counts)
{
    return std::accumulate(counts.begin(), counts.end(), size_type(0));
}

//---------------------------------------------------------------------------//
/*!
 * Do an exclusive scan of the number of surviving secondaries from each track.
 *
 * For an input array x, this calculates the exclusive prefix sum y of the
 * array elements, i.e., \f$ y_i = \sum_{j=0}^{i-1} x_j \f$,
 * where \f$ y_0 = 0 \f$, and stores the result in the input array.
 */
template<>
void exclusive_scan_counts<MemSpace::host>(Span<size_type> counts)
{
    // TODO: Use std::exclusive_scan when C++17 is adopted
    size_type acc = 0;
    for (auto& count_i : counts)
    {
        size_type current = count_i;
        count_i           = acc;
        acc += current;
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
