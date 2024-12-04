//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ParticleTallyData.hh
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
struct ParticleTallyParamsData
{
    //// DATA ////

    //! Number of tally bins
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
    ParticleTallyParamsData&
    operator=(ParticleTallyParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        num_bins = other.num_bins;
        num_particles = other.num_particles;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * State data for accumulating results for each particle type.
 *
 * \c counts is indexed as particle_id * num_bins + bin_index.
 */
template<Ownership W, MemSpace M>
struct ParticleTallyStateData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    Items<size_type> counts;

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return counts.size(); }

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return !counts.empty(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    ParticleTallyStateData& operator=(ParticleTallyStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        counts = other.counts;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
// Resize based on number of bins and particle types
template<MemSpace M>
void resize(ParticleTallyStateData<Ownership::value, M>* state,
            HostCRef<ParticleTallyParamsData> const& params,
            StreamId,
            size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
