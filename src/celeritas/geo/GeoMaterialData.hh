//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterialData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared data for mapping geometry to materials.
 */
template<Ownership W, MemSpace M>
struct GeoMaterialParamsData
{
    template<class T>
    using VolumeItems = celeritas::Collection<T, W, M, VolumeId>;

    VolumeItems<PhysMatId> materials;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !materials.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GeoMaterialParamsData& operator=(GeoMaterialParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        materials = other.materials;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
