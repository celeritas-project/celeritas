//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/cont/EnumArray.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/optical/Types.hh"

#include "BoundaryAction.hh"
#include "SurfaceModel.hh"
#include "SurfacePhysicsData.hh"
#include "SurfaceSteppingAction.hh"

namespace celeritas
{
class ActionRegistry;

namespace inp
{
struct SurfacePhysics;
}  // namespace inp

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Manage properties for optical surface physics.
 *
 * Surface physics during boundary crossing is split into three phases:
 *
 *  1. Initialize boundary crossing
 *  2. Surface physics stepping
 *  3. Post boundary crossing
 *
 * When a surface is crossed in the geometry traversal, the \c
 * InitBoundaryAction is called which initializes the surface physics state for
 * the track. The standard stepping loop is replaced with the surface physics
 * stepping action which calls each surface physics model in appropriate order.
 * When the track is leaving the surface, the \c PostBoundaryAction is called
 * to clean up the state and update the geometry.
 */
class SurfacePhysicsParams final
    : public ParamsDataInterface<SurfacePhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<SurfaceModel>;
    using SurfaceStepModels
        = EnumArray<SurfacePhysicsOrder, std::vector<SPModel>>;
    //!@}

  public:
    // Construct surface physics from input
    explicit SurfacePhysicsParams(ActionRegistry* action_reg,
                                  inp::SurfacePhysics const& input);

    //! Access surface physics data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access surface physics data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    //! Action ID for initializing boundary interactions
    ActionId init_boundary_action() const
    {
        return init_boundary_action_->action_id();
    }

    //! Action ID for surface stepping loop action
    ActionId surface_stepping_action() const
    {
        return surface_stepping_action_->action_id();
    }

    //! Action ID for finishing boundary interactions
    ActionId post_boundary_action() const
    {
        return post_boundary_action_->action_id();
    }

    //! Get models for a given sub-step
    std::vector<SPModel> const& models(SurfacePhysicsOrder step) const
    {
        return models_[step];
    }

  private:
    // Boundary actions
    std::shared_ptr<InitBoundaryAction> init_boundary_action_;
    std::shared_ptr<PostBoundaryAction> post_boundary_action_;
    std::shared_ptr<SurfaceSteppingAction> surface_stepping_action_;

    SurfaceStepModels models_;

    // Host/device storage
    ParamsDataStore<SurfacePhysicsParamsData> data_;

    // Build surface data
    void build_surfaces(std::vector<std::vector<OptMatId>> const&,
                        HostVal<SurfacePhysicsParamsData>&) const;

    // Build sub-step models
    SurfaceStepModels build_models(inp::SurfacePhysics const&,
                                   HostVal<SurfacePhysicsParamsData>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
