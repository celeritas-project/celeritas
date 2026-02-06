//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"

#include "SurfaceData.hh"

namespace celeritas
{
namespace inp
{
struct Surfaces;
}
class VolumeParams;

//---------------------------------------------------------------------------//
/*!
 * Map volumetric geometry information to surface IDs.
 *
 * See the introduction to \rstref{the Geometry API section,api_geometry} for
 * a detailed description of surfaces in the detector geometry description.
 *
 * The specification of \em surfaces using \em volume relationships is required
 * by volume-based geometries such as Geant4 and VecGeom 1, so it is not
 * currently possible to define different properties for the different \em
 * faces of a volume unless those faces are surrounded by distinct geometric
 * volumes. Since ORANGE and VecGeom 2 support true surface
 * definitions, a future extension will allow the user to attach surface
 * properties to, for example, different sides of a cube.
 *
 * \internal Construction requirements:
 * - Volumes and instances in the surface input must be within bounds.
 * - Volumes are allowed to be empty if no surfaces are defined.
 */
class SurfaceParams final : public ParamsDataInterface<SurfaceParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceMap = LabelIdMultiMap<SurfaceId>;
    //!@}

  public:
    // Construct from surface input
    SurfaceParams(inp::Surfaces const&, VolumeParams const& volumes);

    // Construct without building any surface data structures
    SurfaceParams();

    //// METADATA ACCESS ////

    //! Whether any surfaces are present
    bool empty() const { return labels_.empty(); }

    //! Whether surfaces are disabled for non-optical problems
    bool disabled() const { return this->host_ref().volume_surfaces.empty(); }

    //! Number of surfaces
    SurfaceId::size_type num_surfaces() const { return labels_.size(); }

    //! Get surface metadata
    SurfaceMap const& labels() const { return labels_; }

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Reference to managed GPU geometry data
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    ParamsDataStore<SurfaceParamsData> data_;
    // Metadata: surface labels
    SurfaceMap labels_;
};

//---------------------------------------------------------------------------//

extern template class ParamsDataStore<SurfaceParamsData>;
extern template class ParamsDataInterface<SurfaceParamsData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
