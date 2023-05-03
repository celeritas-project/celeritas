//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnosticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared diagnostic attributes.
 */
template<Ownership W, MemSpace M>
struct StepDiagnosticParamsData
{
    //// DATA ////

    //! Number of bins in the histogram
    size_type num_bins{0};
    //! Number of particle types
    size_type num_particles{0};

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_bins > 0 && num_particles > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StepDiagnosticParamsData&
    operator=(StepDiagnosticParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        num_bins = other.num_bins;
        num_particles = other.num_particles;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Distribution of steps per track for each particle type.
 *
 * \counts is indexed as particle_id * num_bins + num_steps.
 */
template<Ownership W, MemSpace M>
struct StepDiagnosticStateData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    //! Bin tracks by particle and step count
    Items<size_type> counts;

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return counts.size(); }

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return !counts.empty(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    StepDiagnosticStateData& operator=(StepDiagnosticStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        counts = other.counts;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
// Resize based on number of actions and particle types
template<MemSpace M>
void resize(StepDiagnosticStateData<Ownership::value, M>* state,
            HostCRef<StepDiagnosticParamsData> const& params,
            StreamId,
            size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
