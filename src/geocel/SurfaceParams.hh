//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

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

    // Construct without surfaces
    SurfaceParams();

    //// METADATA ACCESS ////

    //! Whether any surfaces are present
    bool empty() const { return !static_cast<bool>(data_); }

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
    CollectionMirror<SurfaceParamsData> data_;
    // Metadata: surface labels
    SurfaceMap labels_;
};

//---------------------------------------------------------------------------//

extern template class CollectionMirror<SurfaceParamsData>;
extern template class ParamsDataInterface<SurfaceParamsData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
