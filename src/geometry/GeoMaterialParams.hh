//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterialParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/CollectionMirror.hh"
#include "geometry/GeoParams.hh"
#include "physics/material/MaterialParams.hh"
#include "GeoMaterialData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map a track's geometry state to a material ID.
 *
 * For the forseeable future this class should just be a vector of MaterialIds,
 * one per volume.
 *
 * The constructor takes an array of material IDs for every volume. Missing
 * material IDs may be allowed if they correspond to unreachable volume IDs. If
 * the list of `volume_names` strings is provided, it must be the same size as
 * `volume_to_mat` and indicate a mapping for the geometry's volume IDs.
 * Otherwise, the array is required to have exactly one entry per volume ID.
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

    //! Input parameters
    struct Input
    {
        std::shared_ptr<const GeoParams>      geometry;
        std::shared_ptr<const MaterialParams> materials;
        std::vector<MaterialId>               volume_to_mat;
        std::vector<std::string>              volume_names; // Optional
    };

  public:
    // Construct from geometry and material params
    explicit GeoMaterialParams(Input);

    //! Access material properties on the host
    const HostRef& host_ref() const { return data_.host(); }

    //! Access material properties on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    CollectionMirror<GeoMaterialParamsData> data_;

    using HostValue = GeoMaterialParamsData<Ownership::value, MemSpace::host>;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
