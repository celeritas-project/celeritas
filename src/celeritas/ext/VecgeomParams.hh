//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>

#include "corecel/Types.hh"

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
    using HostRef
        = VecgeomParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = VecgeomParamsData<Ownership::const_reference, MemSpace::device>;
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
    const std::string& id_to_label(VolumeId vol_id) const;

    // Get the volume ID corresponding to a label
    VolumeId find_volume(const std::string& label) const;

    //! Maximum nested geometry depth
    int max_depth() const { return host_ref_.max_depth; }

    //// DATA ACCESS ////

    //! Access geometry data on host
    inline const HostRef& host_ref() const;

    //! Access geometry data on host
    inline const DeviceRef& device_ref() const;

  private:
    //// DATA ////

    // Host metadata/access
    std::vector<std::string>                  vol_labels_;
    std::unordered_map<std::string, VolumeId> vol_ids_;

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
