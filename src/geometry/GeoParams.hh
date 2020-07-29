//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGParams_hh
#define geometry_VGParams_hh

#include <string>
#include "Types.hh"
#include "base/Types.hh"
#include "GeoParamsPointers.hh"

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
    // Construct from a GDML filename
    explicit GeoParams(const char* gdml_filename);

    // Clean up VecGeom on destruction
    ~GeoParams();

    // >>> HOST ACCESSORS

    // Get the label for a placed volume ID
    const std::string& id_to_label(VolumeId vol_id) const;

    // Get the ID corresponding to a label
    VolumeId label_to_id(const std::string& label) const;

    //! Number of volumes
    size_type num_volumes() const { return num_volumes_; }

    //! Maximum nested geometry depth
    int max_depth() const { return max_depth_; }

    // View in-host geometry data for CPU debugging
    GeoParamsPointers host_view() const;

    // >>> DEVICE ACCESSORS

    // Get a view to the managed on-device data
    GeoParamsPointers device_pointers() const;

  private:
    int       max_depth_   = 0;
    size_type num_volumes_ = 0;

    const void* device_world_volume_ = nullptr;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geometry_VGParams_hh
