//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DiagnosticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Persistent diagnostic data.
 */
template<Ownership W, MemSpace M>
struct DiagnosticParamsData
{
    //// DATA ////

    bool field_diagnostic{false};

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DiagnosticParamsData& operator=(DiagnosticParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        field_diagnostic = other.field_diagnostic;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Dynamic diagnostic state data.
 *
 * This should *only* be used for state data that is impossible to collect in
 * the actual diagnostic (e.g., pre-step track energy that must be cached for
 * use by a post-step diagnostic). The data will only be allocated if the
 * diagnostic that requires it is enabled.
 */
template<Ownership W, MemSpace M>
struct DiagnosticStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    Items<units::MevEnergy> pre_step_energy;  //!< Pre-step energy
    Items<size_type> num_field_substeps;  //!< Number of field substeps

    //// METHODS ////

    //! Check whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return pre_step_energy.size() == num_field_substeps.size();
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return pre_step_energy.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DiagnosticStateData& operator=(DiagnosticStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        pre_step_energy = other.pre_step_energy;
        num_field_substeps = other.num_field_substeps;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize diagnostic data.
 */
template<MemSpace M>
void resize(DiagnosticStateData<Ownership::value, M>* state,
            HostCRef<DiagnosticParamsData> const& params,
            size_type size)
{
    CELER_EXPECT(size > 0);
    if (params.field_diagnostic)
    {
        resize(&state->pre_step_energy, size);
        resize(&state->num_field_substeps, size);
        fill(size_type{0}, &state->num_field_substeps);
        CELER_ASSERT(state->size() == size);
    }
    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
