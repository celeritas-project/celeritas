//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geom/VGHost.h
//---------------------------------------------------------------------------//
#ifndef geom_VGGeometryHost_h
#define geom_VGGeometryHost_h

#include <string>
#include "Types.hh"
#include "base/Types.hh"

namespace celeritas
{
struct VGView;
//---------------------------------------------------------------------------//
/*!
 * Wrap a VecGeom geometry definition with convenience functions.
 */
class VGHost
{
  public:
    // Construct from a GDML filename
    explicit VGHost(const char* gdml_filename);

    // >>> ACCESSORS

    // Get the label for a placed volume ID
    const std::string& id_to_label(VolumeId vol_id) const;
    // Get the ID corresponding to a label
    VolumeId label_to_id(const std::string& label) const;

    //! Number of volumes
    size_type num_volumes() const { return num_volumes_; }

    //! Maximum nested geometry depth
    int max_depth() const { return max_depth_; }

    // View in-host geometry data for CPU debugging
    VGView host_view() const;

  private:
    int       max_depth_;
    size_type num_volumes_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geom_VGGeometryHost_h
