//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecgeomParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>
#include "base/Types.hh"
#include "geometry/Types.hh"
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

    //! View in-host geometry data for CPU debugging
    const HostRef& host_ref() const { return host_ref_; }

    //! Get a view to the managed on-device data
    const DeviceRef& device_ref() const { return device_ref_; }

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
} // namespace celeritas
