//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleCaloData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Number of detectors being tallied.
template<Ownership W, MemSpace M>
struct SimpleCaloParamsData
{
    DetectorId::size_type num_detectors{0};

    explicit CELER_FUNCTION operator bool() const { return num_detectors > 0; }

    template<Ownership W2, MemSpace M2>
    SimpleCaloParamsData& operator=(SimpleCaloParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        num_detectors = other.num_detectors;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Accumulated calorimeter data for a set of tracks.
 *
 * This should be specific to a single StreamId but will be integrated over all
 * tracks on that stream. Depending on whether the goal is to obtain a
 * statistical average of energy deposition, or to find energy deposition for a
 * particular event, the detector data can be reset at the end of each event,
 * at the end of a batch of events, or at the end of the simulation.
 */
template<Ownership W, MemSpace M>
struct SimpleCaloStateData
{
    //// TYPES ////

    template<class T>
    using DetItems = celeritas::Collection<T, W, M, DetectorId>;
    using EnergyUnits = units::Mev;

    //// DATA ////

    // Energy indexed by detector ID
    DetItems<real_type> energy_deposition;

    // Number of track slots (unused during calculation)
    size_type num_track_slots{};

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return num_track_slots; }

    //! True if constructed
    explicit CELER_FUNCTION operator bool() const
    {
        return !energy_deposition.empty() && num_track_slots > 0;
    }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    SimpleCaloStateData& operator=(SimpleCaloStateData<W2, M2>& other)
    {
        energy_deposition = other.energy_deposition;
        num_track_slots = other.num_track_slots;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
// Resize based on the number of detectors
template<MemSpace M>
void resize(SimpleCaloStateData<Ownership::value, M>* state,
            HostCRef<SimpleCaloParamsData> const& params,
            StreamId,
            size_type num_track_slots);

//---------------------------------------------------------------------------//
}  // namespace celeritas
