//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/Span.hh"

#include "VecgeomData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared model parameters for a VecGeom geometry.
 *
 * The model defines the shapes, volumes, etc.
 */
class VecgeomParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef           = HostCRef<VecgeomParamsData>;
    using DeviceRef         = DeviceCRef<VecgeomParamsData>;
    using SpanConstVolumeId = Span<const VolumeId>;
    //!@}

  public:
    // Construct from a GDML filename
    explicit VecgeomParams(const std::string& gdml_filename);

    // Clean up VecGeom on destruction
    ~VecgeomParams();

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const { return true; }

    //// VOLUMES ////

    //! Number of volumes
    VolumeId::size_type num_volumes() const { return vol_labels_.size(); }

    // Get the label for a placed volume ID
    const Label& id_to_label(VolumeId vol_id) const;

    // Get the volume ID corresponding to a unique label name
    VolumeId find_volume(const std::string& name) const;

    // Get zero or more volume IDs corresponding to a name
    SpanConstVolumeId find_volumes(const std::string& name) const;

    //! Maximum nested geometry depth
    int max_depth() const { return host_ref_.max_depth; }

    //// SURFACES (NOT APPLICABLE FOR VECGEOM) ////

    // Get the label for a placed volume ID
    inline const Label& id_to_label(SurfaceId) const;

    // Get the surface ID corresponding to a unique label name
    inline SurfaceId find_surface(const std::string& name) const;

    //! Number of distinct surfaces
    size_type num_surfaces() const { return 0; }

    //// DATA ACCESS ////

    //! Access geometry data on host
    inline const HostRef& host_ref() const;

    //! Access geometry data on host
    inline const DeviceRef& device_ref() const;

  private:
    //// DATA ////

    // Host metadata/access
    LabelIdMultiMap<VolumeId> vol_labels_;

    // Host/device storage and reference
    HostRef   host_ref_;
    DeviceRef device_ref_;

    //// HELPER FUNCTIONS ////

    void build_md();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * No surface IDs are defined in vecgeom.
 */
const Label& VecgeomParams::id_to_label(SurfaceId) const
{
    CELER_NOT_IMPLEMENTED("surfaces in VecGeom");
}

//---------------------------------------------------------------------------//
/*!
 * No surface IDs are defined in vecgeom.
 */
SurfaceId VecgeomParams::find_surface(const std::string&) const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Access geometry data on host.
 */
auto VecgeomParams::host_ref() const -> const HostRef&
{
    CELER_ENSURE(host_ref_);
    return host_ref_;
}

//---------------------------------------------------------------------------//
/*!
 * Access geometry data on device.
 */
auto VecgeomParams::device_ref() const -> const DeviceRef&
{
    CELER_ENSURE(device_ref_);
    return device_ref_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
