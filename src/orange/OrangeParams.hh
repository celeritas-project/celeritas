//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Label.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionMirror.hh"

#include "BoundingBox.hh"
#include "Data.hh"
#include "Types.hh"
#include "detail/UnitIndexer.hh"

namespace celeritas
{
struct OrangeInput;
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
    using HostRef           = HostCRef<OrangeParamsData>;
    using DeviceRef         = DeviceCRef<OrangeParamsData>;
    using SpanConstVolumeId = Span<const VolumeId>;
    //!@}

  public:
    // Construct from a JSON file (if JSON is enabled)
    explicit OrangeParams(const std::string& json_filename);

    // ADVANCED usage: construct from explicit host data
    explicit OrangeParams(OrangeInput input);

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const { return supports_safety_; }

    //// VOLUMES ////

    //! Number of volumes
    VolumeId::size_type num_volumes() const { return vol_labels_.size(); }

    // Get the label for a placed volume ID
    const Label& id_to_label(VolumeId vol_id) const;

    // Get the volume ID corresponding to a unique label
    VolumeId find_volume(const std::string& name) const;

    // Get zero or more volume IDs corresponding to a name
    SpanConstVolumeId find_volumes(const std::string& name) const;

    //! Outer bounding box of geometry
    const BoundingBox& bbox() const { return bbox_; }

    //// SURFACES ////

    // Get the label for a placed volume ID
    const Label& id_to_label(SurfaceId surf_id) const;

    // Get the surface ID corresponding to a unique label name
    SurfaceId find_surface(const std::string& name) const;

    //! Number of distinct surfaces
    SurfaceId::size_type num_surfaces() const { return surf_labels_.size(); }

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    const HostRef& host_ref() const { return data_.host(); }

    //! Reference to managed GPU geometry data
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    // Host metadata/access
    detail::UnitIndexer        unit_indexer_;
    LabelIdMultiMap<SurfaceId> surf_labels_;
    LabelIdMultiMap<VolumeId>  vol_labels_;
    BoundingBox                bbox_;
    bool                       supports_safety_{};

    // Host/device storage and reference
    CollectionMirror<OrangeParamsData> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
