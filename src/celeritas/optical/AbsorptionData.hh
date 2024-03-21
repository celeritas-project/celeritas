//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionData.hh
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
 * Shared absorption properties data.
 */
template<Ownership W, Memspace M>
struct AbsorptionData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;

    //// MEMBER DATA ////

    OpticalMaterialItems<GenericGridData> absorption_length;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !absorption_length.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    AbsorptionData& operator=(AbsorptionData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        absorption_length = other.absorption_length;
        reals = other.reals;
        return *this;
    }
};

using AbsorptionDeviceRef = DeviceCRef<AbsorptionData>;
using AbsorptionHostRef = HostCRef<AbsorptionData>;
using AbsorptionRef = NativeCRef<AbsorptionData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
