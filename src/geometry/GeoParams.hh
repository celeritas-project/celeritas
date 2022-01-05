//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include "base/Types.hh"
#include "GeoData.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared model parameters for a VecGeom geometry.
 *
 * The model defines the shapes, volumes, etc.
 */
class GeoParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef = GeoParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = GeoParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

  public:
    // Construct from a GDML filename
    explicit GeoParams(const std::string& gdml_filename);

    // Clean up VecGeom on destruction
    ~GeoParams();

    //// HOST ACCESSORS ////

    // Get the label for a placed volume ID
    const std::string& id_to_label(VolumeId vol_id) const;

    // Get the ID corresponding to a label
    // TODO: rename to find ?? see MaterialParams etc.
    VolumeId label_to_id(const std::string& label) const;

    //! Number of volumes
    VolumeId::size_type num_volumes() const { return num_volumes_; }

    //! Maximum nested geometry depth
    int max_depth() const { return host_ref_.max_depth; }

    //! View in-host geometry data for CPU debugging
    const HostRef& host_ref() const { return host_ref_; }

    //! Get a view to the managed on-device data
    const DeviceRef& device_ref() const { return device_ref_; }

  private:
    size_type num_volumes_ = 0;

    HostRef   host_ref_;
    DeviceRef device_ref_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
