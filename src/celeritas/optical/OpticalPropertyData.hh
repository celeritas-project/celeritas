//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPropertyData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared optical properties data.
 *
 * TODO: Placeholder for optical property data; modify or replace as needed.
 */
template<Ownership W, MemSpace M>
struct OpticalPropertyData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;

    //// MEMBER DATA ////

    Items<real_type> reals;
    OpticalMaterialItems<GenericGridData> refractive_index;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !refractive_index.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalPropertyData& operator=(OpticalPropertyData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        refractive_index = other.refractive_index;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
