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
struct OpticalSurfacePhysics;
}  // namespace inp

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Manage properties for optical surface physics.
 *
 * Surface physics during boundary crossing is split into three actions.
 *
 *  1. Initialize the boundary crossing
 *     (see \rstref{Boundary initialization,surface_boundary_init} ).
 *  2. Apply surface physics models.
 *  3. If exiting the surface, the crossing or move into the previous volume
 *     (see \rstref{Boundary crossing,surface_boundary_post} ).
 *
 * The second action is broken down into sets of "internal" \c
 * celeritas::optical::SurfaceModel classes categorized by a
 * \c SurfacePhysicsOrder . These are called sequentially, with one model from
 * each set applying to each track on a surface.
 *
 *  1. Sample a local facet normal
 *     (see \rstref{Roughness,surface_roughness} ).
 *  2. Select transmittance/reflectivity/absorption override
 *     (see \rstref{Reflectivity,surface_reflectivity} ).
 *  3. Sample and perform reflection/refraction
 *     (see \rstref{Interaction,surface_interaction} ).
 *
 * \note The developer documentation includes further details of the
 * low-level \c Executor classes that performs these actions.
 *
 * \internal See the user manual for high-level descriptions and low-level
 * samplers. The intermediate action/executor descriptions follow.
 *
 * The three actions from the surface physics params are:
 * 1. \c BoundaryAction instantiated to use \c detail::InitBoundaryExecutor
 * 2. \c SurfaceSteppingAction, which invokes the surface model kernels.
 * 3. \c BoundaryAction instantiated to use \c detail::PostBoundaryExecutor
 *
 * Each surface model launches a kernel:
 * - Gaussian roughness (\c detail::GaussianRoughnessExecutor which uses \c
 *    GaussianRoughnessSampler and rejects with \c EnteringSurfaceNormalSampler
 * - Smear roughness (\c detail::SmearRoughnessExectuor)
 * - Grid reflectivity \c detail::GridReflectivityExecutor
 * - Dielectric interaciton, wrapping \c detail::DielectricInteractor with \c
 *   SurfaceInteractionApplier .
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
                                  inp::OpticalSurfacePhysics const& input);

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
    std::shared_ptr<SurfaceSteppingAction> surface_stepping_action_;
    std::shared_ptr<PostBoundaryAction> post_boundary_action_;

    SurfaceStepModels models_;

    // Host/device storage
    ParamsDataStore<SurfacePhysicsParamsData> data_;

    // Build surface data
    void build_surfaces(std::vector<std::vector<OptMatId>> const&,
                        HostVal<SurfacePhysicsParamsData>&) const;

    // Build sub-step models
    SurfaceStepModels build_models(inp::OpticalSurfacePhysics const&,
                                   HostVal<SurfacePhysicsParamsData>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
