//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.cc
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include "InitTracksLauncher.hh"
#include "LocateAliveLauncher.hh"
#include "ProcessPrimariesLauncher.hh"
#include "ProcessSecondariesLauncher.hh"

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
    auto num_vacancies = min(data.vacancies.size(), data.initializers.size());

    InitTracksLauncher<MemSpace::host> launch(params, states, data);
#pragma omp parallel for
    for (size_type i = 0; i < num_vacancies; ++i)
    {
        launch(ThreadId{i});
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
#pragma omp parallel for
    for (size_type i = 0; i < states.size(); ++i)
    {
        launch(ThreadId{i});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers on host from primary particles.
 */
void process_primaries(Span<const Primary>          primaries,
                       const TrackInitStateHostRef& data)
{
    ProcessPrimariesLauncher<MemSpace::host> launch(primaries, data);
#pragma omp parallel for
    for (size_type i = 0; i < primaries.size(); ++i)
    {
        launch(ThreadId{i});
    }
}
//---------------------------------------------------------------------------//
/*!
 * Create track initializers on host from secondary particles.
 */
void process_secondaries(const ParamsHostRef&         params,
                         const StateHostRef&          states,
                         const TrackInitStateHostRef& data)
{
    ProcessSecondariesLauncher<MemSpace::host> launch(params, states, data);
#pragma omp parallel for
    for (size_type i = 0; i < states.size(); ++i)
    {
        launch(ThreadId{i});
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
