//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterialData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "geometry/Types.hh"
#include "physics/material/Types.hh"

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

    VolumeItems<MaterialId> materials;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !materials.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GeoMaterialParamsData& operator=(const GeoMaterialParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        materials = other.materials;
        return *this;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
