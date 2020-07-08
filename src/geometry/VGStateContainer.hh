//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateContainer.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGStateContainer_hh
#define geometry_VGStateContainer_hh

#include <memory>
#include "VGHost.hh"
#include "base/Types.hh"

namespace celeritas
{
class VGDevice;
class VGStateContainerPimpl;
struct VGStateView;
//---------------------------------------------------------------------------//
/*!
 * Manage on-device VecGeom geometry and states.
 */
class VGStateContainer
{
  public:
    //@{
    //! Type aliases
    using constSPVGDevice = std::shared_ptr<const VGDevice>;
    //@}

  public:
    // Construct from device geometry and number of track states
    VGStateContainer(constSPVGDevice device_geom, size_type size);

    //@{
    //! Defaults that cause thrust to launch kernels
    VGStateContainer();
    ~VGStateContainer();
    VGStateContainer(VGStateContainer&&);
    VGStateContainer& operator=(VGStateContainer&&);
    //@}

    // >>> ACCESSORS

    // Number of states
    size_type size() const { return state_size_; }

    // View on-device states
    VGStateView device_view() const;

  private:
    constSPVGDevice device_geom_;
    size_type       state_size_ = 0;

    std::unique_ptr<VGStateContainerPimpl> state_data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geometry_VGStateContainer_hh
