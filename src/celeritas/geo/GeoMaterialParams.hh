//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * For the foreseeable future this class should just be a vector of
 * PhysicsMaterialIds, one per volume.
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
    using SPConstCoreGeo = std::shared_ptr<CoreGeoParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    //!@}

    //! Input parameters
    struct Input
    {
        SPConstCoreGeo geometry;
        SPConstMaterial materials;
        std::vector<PhysMatId> volume_to_mat;
        std::vector<Label> volume_labels;  // Optional
    };

  public:
    // Construct with imported data
    static std::shared_ptr<GeoMaterialParams>
    from_import(ImportData const& data,
                SPConstCoreGeo geo_params,
                SPConstMaterial material_params);

    // Construct from geometry and material params
    explicit GeoMaterialParams(Input);

    //! Access material properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access material properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    // Get the total number of volumes
    inline VolumeId::size_type num_volumes() const;

    // Get the material ID corresponding to a volume ID
    inline PhysMatId material_id(VolumeId v) const;

  private:
    CollectionMirror<GeoMaterialParamsData> data_;

    using HostValue = HostVal<GeoMaterialParamsData>;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the total number of volumes.
 */
VolumeId::size_type GeoMaterialParams::num_volumes() const
{
    return this->host_ref().materials.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the material ID corresponding to a volume ID.
 *
 * Some "virtual" volumes may have a null ID.
 */
PhysMatId GeoMaterialParams::material_id(VolumeId v) const
{
    CELER_EXPECT(v < this->num_volumes());

    return this->host_ref().materials[v];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
