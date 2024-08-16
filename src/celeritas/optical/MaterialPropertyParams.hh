//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MaterialPropertyParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

#include "MaterialPropertyData.hh"

namespace celeritas
{
struct ImportData;

namespace optical
{

//---------------------------------------------------------------------------//
/*!
 * Manage properties for optical materials.
 *
 * More than one "geometry material" (i.e. \c G4Material or material defined in
 * the geometry model input) can point to a single optical material.
 *
 * Optical volume and surface properties are imported from Geant4 into the \c
 * ImportData container. The \c celeritas::MaterialParams class loads the
 * mapping of \c GeoMaterialId to \c OpticalMaterialId and makes it accessible
 * via the main loop's material view.
 */
class MaterialPropertyParams final
    : public ParamsDataInterface<MaterialPropertyData>
{
  public:
    // Shared optical properties, indexed by \c OpticalMaterialId
    struct Input
    {
        std::vector<ImportOpticalProperty> data;
    };

  public:
    // Construct with imported data
    static std::shared_ptr<MaterialPropertyParams>
    from_import(ImportData const& data);

    // Construct with optical property data
    explicit MaterialPropertyParams(Input const& inp);

    //! Access optical properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    CollectionMirror<MaterialPropertyData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
