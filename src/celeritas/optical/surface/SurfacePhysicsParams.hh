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
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "BoundaryAction.hh"
#include "SurfaceModel.hh"
#include "SurfacePhysicsData.hh"

namespace celeritas
{
class ActionRegistry;

namespace optical
{
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
    template<class T>
    using SurfaceStepArray = EnumArray<SurfacePhysicsStep, T>;

    using SPModel = std::shared_ptr<SurfaceModel>;
    using VecModelBuilders = std::vector<SurfaceModel::ModelBuilder>;
    //!@}

    struct SurfaceInput
    {
        std::vector<OptMatId> materials;
        std::vector<SurfaceStepArray<SurfaceModelId>> interface_models;
    };

    struct Input
    {
        ActionRegistry* action_reg = nullptr;

        std::vector<SurfaceInput> surfaces;  //!< indexed by GeometricSurfaceId
        SurfaceStepArray<VecModelBuilders> model_builders;
    };

  public:
    explicit SurfacePhysicsParams(Input);

    //! Access surface physics data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access surface physics data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    //! Action ID for initializing boundary interactions
    ActionId init_boundary_action() const
    {
        return init_boundary_action_->action_id();
    }

    //! Action ID for finishing boundary interactions
    ActionId post_boundary_action() const
    {
        return post_boundary_action_->action_id();
    }

    //! Get models for a given sub-step
    std::vector<SPModel> const& models(SurfacePhysicsStep step) const
    {
        return models_[step];
    }

  private:
    // Boundary actions
    std::shared_ptr<InitBoundaryAction> init_boundary_action_;
    std::shared_ptr<PostBoundaryAction> post_boundary_action_;

    SurfaceStepArray<std::vector<SPModel>> models_;

    // Host/device storage
    CollectionMirror<SurfacePhysicsParamsData> data_;

    // Build sub-step models
    SurfaceStepArray<std::vector<SPModel>>
    build_models(SurfaceStepArray<VecModelBuilders> const&) const;

    // Build surface data
    void build_surfaces(std::vector<SurfaceInput> const&,
                        HostVal<SurfacePhysicsParamsData>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
