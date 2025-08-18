//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/phys/SurfaceModel.hh"

#include "SurfacePhysicsData.hh"

namespace celeritas
{
namespace optical
{

// class SurfaceModel : public ::celeritas::SurfaceModel, public ConcreteAction
// {
// };

//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfacePhysicsParams ...;
   \endcode
 */
class SurfacePhysicsParams final
    : public ParamsDataInterface<SurfacePhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    explicit SurfacePhysicsParams() {}

    //! Access surface physics data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access surface physics data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    //! Action ID for initializing boundary interactions
    ActionId init_boundary_action() const { return ActionId{}; }

    //! Action ID for finishing boundary interactions
    ActionId post_boundary_action() const { return ActionId{}; }

    std::vector<std::shared_ptr<SurfaceModel>> models(SurfacePhysicsStep) const
    {
        return {};
    }

  private:
    // Host/device storage
    CollectionMirror<SurfacePhysicsParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
