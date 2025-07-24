//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/optical/action/ActionInterface.hh"

#include "InitBoundaryAction.hh"
#include "PostBoundaryAction.hh"
#include "SurfacePhysicsData.hh"

namespace celeritas
{
class ActionRegistry;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class SurfacePhysicsParams final
    : public ParamsDataInterface<SurfacePhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using ActionIdRange = Range<ActionId>;
    //!@}

    struct Input
    {
        ActionRegistry* action_registry = nullptr;
    };

  public:
    // Construct from models
    explicit SurfacePhysicsParams(Input input);

    //! Access surface physics data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access surface physics data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    //! Action ID for initializing boundary interactions
    ActionId init_boundary_action() const
    {
        return this->init_boundary_action_->action_id();
    }

    //! Action ID for finishing boundary interactions
    ActionId post_boundary_action() const
    {
        return this->post_boundary_action_->action_id();
    }

  private:
    // Actions
    std::shared_ptr<InitBoundaryAction> init_boundary_action_;
    std::shared_ptr<PostBoundaryAction> post_boundary_action_;

    // Host/device storage
    CollectionMirror<SurfacePhysicsParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
