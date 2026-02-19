//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MaterialParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

#include "MaterialData.hh"
#include "MaterialView.hh"

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
 * Each "physics material" (the combination of a geometry-specified material
 * and a user-specified region) can map to a single optical material. Many
 * materials---especially those in mechanical structures and components not
 * optically connected to the detector---may have no optical properties at all.
 *
 * Optical volume properties are imported from Geant4 into the \c
 * ImportData container. The \c celeritas::MaterialParams class loads the
 * mapping of \c PhysMatId to \c OptMatId and makes it
 * accessible via the main loop's material view. By combining that with the \c
 * GeoMaterialParams which maps volumes to \c PhysMatId, this class
 * maps the geometry volumes to optical materials for use in the optical
 * tracking loop.
 *
 * When surface models are implemented, surface properties will also be added
 * to this class.
 */
class MaterialParams final : public ParamsDataInterface<MaterialParamsData>
{
  public:
    struct Input
    {
        //! Shared optical material, indexed by \c OptMatId
        std::vector<ImportOpticalProperty> properties;
        //! Map logical volume ID to optical material ID
        std::vector<OptMatId> volume_to_mat;
        //! Map optical material ID to core material ID
        std::vector<PhysMatId> optical_to_core;
    };

  public:
    // Construct with imported data, materials
    static std::shared_ptr<MaterialParams>
    from_import(ImportData const& data,
                ::celeritas::GeoMaterialParams const& geo_mat,
                ::celeritas::MaterialParams const& mat);

    // Construct with optical property data
    explicit MaterialParams(Input const& inp);

    // Number of optical materials
    inline OptMatId::size_type num_materials() const;

    // Construct a material view for the given identifier
    MaterialView get(OptMatId mat) const;

    //! Access optical material on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical material on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    ParamsDataStore<MaterialParamsData> data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Number of optical materials.
 */
OptMatId::size_type MaterialParams::num_materials() const
{
    return this->host_ref().refractive_index.size();
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
