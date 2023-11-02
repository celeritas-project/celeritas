//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnosticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/UniformGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared diagnostic attributes.
 */
template<Ownership W, MemSpace M>
struct FieldDiagnosticParamsData
{
    //// DATA ////

    //! Energy grid
    UniformGridData energy;
    //! Number of substep bins
    size_type num_substep_bins{0};

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return energy && num_substep_bins > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    FieldDiagnosticParamsData&
    operator=(FieldDiagnosticParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        energy = other.energy;
        num_substep_bins = other.num_substep_bins;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * State data for recording field substeps vs. track energy.
 *
 * \c counts is indexed as energy_bin_index * num_substep_bins +
 * substep_bin_index.
 */
template<Ownership W, MemSpace M>
struct FieldDiagnosticStateData
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
    FieldDiagnosticStateData& operator=(FieldDiagnosticStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        counts = other.counts;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
// Resize based on number of bins
template<MemSpace M>
void resize(FieldDiagnosticStateData<Ownership::value, M>* state,
            HostCRef<FieldDiagnosticParamsData> const& params,
            StreamId,
            size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
