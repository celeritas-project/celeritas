//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGDevice.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGDevice_hh
#define geometry_VGDevice_hh

#include <memory>
#include "VGHost.hh"
#include "base/Types.hh"

namespace celeritas
{
struct VGView;
//---------------------------------------------------------------------------//
/*!
 * Manage on-device VecGeom geometry and states.
 */
class VGDevice
{
  public:
    //@{
    //! Type aliases
    using constSPVGHost = std::shared_ptr<const VGHost>;
    //@}

  public:
    // Construct from host geometry
    explicit VGDevice(constSPVGHost host_geom);

    // >>> ACCESSORS

    // View on-device data
    VGView device_view() const;

    // Access maximum geometry object depth
    int max_depth() const { return host_geom_->max_depth(); }

  private:
    constSPVGHost host_geom_;
    const void*   device_world_volume_ = nullptr;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geometry_VGDevice_hh
