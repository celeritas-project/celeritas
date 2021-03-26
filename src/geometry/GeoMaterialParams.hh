//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterialParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/CollectionMirror.hh"
#include "geometry/GeoParams.hh"
#include "physics/material/MaterialParams.hh"
#include "GeoMaterialInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map a track's geometry state to a material ID.
 *
 * For the forseeable future this class should just be a vector of MaterialIds,
 * one per volume.
 */
class GeoMaterialParams
{
  public:
    //!@{
    //! Type aliases
    using HostRef
        = GeoMaterialParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = GeoMaterialParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

    struct Input
    {
        std::shared_ptr<const GeoParams>      geometry;
        std::shared_ptr<const MaterialParams> materials;
        std::vector<MaterialId>               volume_to_mat;
    };

  public:
    // Construct from geometry and material params
    explicit GeoMaterialParams(Input);

    //! Access material properties on the host
    const HostRef& host_pointers() const { return data_.host(); }

    //! Access material properties on the device
    const DeviceRef& device_pointers() const { return data_.device(); }

  private:
    CollectionMirror<GeoMaterialParamsData> data_;

    using HostValue = GeoMaterialParamsData<Ownership::value, MemSpace::host>;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
