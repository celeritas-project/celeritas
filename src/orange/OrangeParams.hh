//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OrangeParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "base/CollectionMirror.hh"
#include "base/Types.hh"
#include "geometry/BoundingBox.hh"
#include "Data.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Persistent model data for an ORANGE geometry.
 *
 * This class initializes and manages the data used by ORANGE (surfaces,
 * volumes) and provides a host-based interface for them.
 */
class OrangeParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef
        = OrangeParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = OrangeParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

    struct Input
    {
        using Surfaces = SurfaceData<Ownership::value, MemSpace::host>;
        using Volumes  = VolumeData<Ownership::value, MemSpace::host>;
        using VecStr   = std::vector<std::string>;

        Surfaces    surfaces;       //!< Surface definitions
        Volumes     volumes;        //!< Volume definitions
        VecStr      surface_labels; //!< Surface names (metadata)
        VecStr      volume_labels;  //!< Volume names (metadata)
        BoundingBox bbox;           //!< Outer bounding box (metadata)
    };

  public:
    // Construct from a JSON file (if JSON is enabled)
    explicit OrangeParams(const std::string& json_filename);

    // ADVANCED usage: construct from explicit host data
    explicit OrangeParams(Input input);

    //// VOLUMES ////

    //! Number of volumes
    VolumeId::size_type num_volumes() const { return vol_labels_.size(); }

    // Get the label for a placed volume ID
    const std::string& id_to_label(VolumeId vol_id) const;

    // Get the volume ID corresponding to a label
    VolumeId find_volume(const std::string& label) const;

    //! Outer bounding box of geometry
    const BoundingBox& bbox() const { return bbox_; }

    //// SURFACES ////

    // Get the label for a placed volume ID
    const std::string& id_to_label(SurfaceId surf_id) const;

    // Get the surface ID corresponding to a label
    SurfaceId find_surface(const std::string& label) const;

    //! Number of distinct surfaces
    size_type num_surfaces() const { return surf_labels_.size(); }

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    const HostRef& host_ref() const { return data_.host(); }

    //! Reference to managed GPU geometry data
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    // Host metadata/access
    std::vector<std::string>                   surf_labels_;
    std::vector<std::string>                   vol_labels_;
    std::unordered_map<std::string, VolumeId>  vol_ids_;
    std::unordered_map<std::string, SurfaceId> surf_ids_;
    BoundingBox                                bbox_;

    // Host/device storage and reference
    CollectionMirror<OrangeParamsData> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
