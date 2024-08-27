//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MaterialParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

#include "MaterialData.hh"

namespace celeritas
{
struct ImportData;
class MaterialParams;
class GeoMaterialParams;

namespace optical
{

//---------------------------------------------------------------------------//
/*!
 * Manage properties for optical materials.
 *
 * Each "geometry material" (i.e. \c G4Material or material defined in
 * the geometry model input) can map to a single optical material. (In the
 * future we might broaden this to allow \c regions to define different
 * materials as well.) Many "geometry materials"---especially those in
 * mechanical structures and components not optically connected to the
 * detector---may have no optical properties at all.
 *
 * Optical volume and surface properties are imported from Geant4 into the \c
 * ImportData container. The \c celeritas::MaterialParams class loads the
 * mapping of \c GeoMaterialId to \c OpticalMaterialId and makes it accessible
 * via the main loop's material view. This class maps the geometry volumes to
 * optical materials for use during tracking. When surface models are
 */
class MaterialParams final : public ParamsDataInterface<MaterialParamsData>
{
  public:
    struct Input
    {
        //! Shared optical material, indexed by \c OpticalMaterialId
        std::vector<ImportOpticalProperty> properties;
        //! Map logical volume ID to optical material ID
        std::vector<OpticalMaterialId> volume_to_mat;
    };

  public:
    // Construct with imported data, materials
    static std::shared_ptr<MaterialParams>
    from_import(ImportData const& data,
                ::celeritas::GeoMaterialParams const& geo_mat,
                ::celeritas::MaterialParams const& mat);

    // Construct with optical property data
    explicit MaterialParams(Input const& inp);

    //! Access optical material on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical material on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    CollectionMirror<MaterialParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
