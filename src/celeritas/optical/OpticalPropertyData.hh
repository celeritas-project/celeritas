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
#include "celeritas/grid/XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Optical properties for a single material.
 *
 * TODO: Placeholder for optical property data; modify or replace as needed.
 */
struct OpticalMaterial
{
    using EnergyUnits = units::Mev;

    // Tabulated refractive index as a funtion of photon energy
    GenericGridData refractive_index;

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(refractive_index);
    }
};

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
    using MaterialItems = Collection<T, W, M, MaterialId>;

    //// MEMBER DATA ////

    Items<real_type> reals;
    MaterialItems<OpticalMaterial> materials;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !materials.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalPropertyData& operator=(OpticalPropertyData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        materials = other.materials;
        return *this;
    }
};

using OpticalPropertyDeviceRef = DeviceCRef<OpticalPropertyData>;
using OpticalPropertyHostRef = HostCRef<OpticalPropertyData>;
using OpticalPropertyRef = NativeCRef<OpticalPropertyData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
