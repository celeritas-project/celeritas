//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterialParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/io/Label.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "GeoFwd.hh"
#include "GeoMaterialData.hh"

namespace celeritas
{
struct ImportData;

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
 *
 * Warnings are emitted if materials are unavailable for any volumes, *unless*
 * the corresponding volume name is empty (corresponding perhaps to a "parallel
 * world" or otherwise unused volume) or is enclosed with braces (used for
 * virtual volumes such as `[EXTERIOR]` or temporary boolean/reflected volumes.
 */
class GeoMaterialParams final
    : public ParamsDataInterface<GeoMaterialParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    //!@}

    //! Input parameters
    struct Input
    {
        SPConstGeo geometry;
        SPConstMaterial materials;
        std::vector<MaterialId> volume_to_mat;
        std::vector<Label> volume_labels;  // Optional
    };

  public:
    // Construct with imported data
    static std::shared_ptr<GeoMaterialParams>
    from_import(ImportData const& data,
                SPConstGeo geo_params,
                SPConstMaterial material_params);

    // Construct from geometry and material params
    explicit GeoMaterialParams(Input);

    //! Access material properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access material properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    CollectionMirror<GeoMaterialParamsData> data_;

    using HostValue = HostVal<GeoMaterialParamsData>;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
