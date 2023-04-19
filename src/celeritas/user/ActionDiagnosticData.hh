//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionDiagnosticData.hh
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
struct ActionDiagnosticParamsData
{
    //// DATA ////

    //! Number of actions
    size_type num_actions{0};
    //! Number of particle types
    size_type num_particles{0};

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_actions > 0 && num_particles > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ActionDiagnosticParamsData&
    operator=(ActionDiagnosticParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        num_actions = other.num_actions;
        num_particles = other.num_particles;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Number of occurrances of each action accumulated over tracks.
 *
 * Actions are tallied separately for each particle type. \c counts is indexed
 * as action ID * number of particle types + particle ID.
 */
template<Ownership W, MemSpace M>
struct ActionDiagnosticStateData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    //! Number of occurrances of each action for each particle type
    Items<size_type> counts;

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return counts.size(); }

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return !counts.empty(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    ActionDiagnosticStateData&
    operator=(ActionDiagnosticStateData<W2, M2>& other)
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
void resize(ActionDiagnosticStateData<Ownership::value, M>* state,
            HostCRef<ActionDiagnosticParamsData> const& params,
            StreamId,
            size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
