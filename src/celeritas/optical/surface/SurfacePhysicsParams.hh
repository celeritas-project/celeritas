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
#include "SurfaceModel.hh"
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
    using SPConstRoughnessModel = std::shared_ptr<SurfaceRoughnessModel const>;
    using SPConstReflectivityModel
        = std::shared_ptr<SurfaceReflectivityModel const>;
    using SPConstInteractionModel
        = std::shared_ptr<SurfaceInteractionModel const>;

    using VecRoughnessModels = std::vector<SPConstRoughnessModel>;
    using VecReflectivityModels = std::vector<SPConstReflectivityModel>;
    using VecInteractionModels = std::vector<SPConstInteractionModel>;

    using ActionIdRange = Range<ActionId>;
    //!@}

    struct SurfaceInput
    {
        RoughnessModelId roughness_model;
        ReflectivityModelId reflectivity_model;
        InteractionModelId interaction_model;
    };

    struct Input
    {
        std::vector<SurfaceRoughnessModel::ModelBuilder> roughness_model_builders;
        std::vector<SurfaceReflectivityModel::ModelBuilder>
            reflectivity_model_builders;
        std::vector<SurfaceInteractionModel::ModelBuilder>
            interaction_model_builders;

        std::vector<SurfaceInput> surfaces;
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

    VecRoughnessModels const& roughness_models() const
    {
        return roughness_models_;
    }
    VecReflectivityModels const& reflectivity_models() const
    {
        return reflectivity_models_;
    }
    VecInteractionModels const& interaction_models() const
    {
        return interaction_models_;
    }

  private:
    // Actions
    VecRoughnessModels roughness_models_;
    VecReflectivityModels reflectivity_models_;
    VecInteractionModels interaction_models_;

    std::shared_ptr<InitBoundaryAction> init_boundary_action_;

    // Host/device storage
    CollectionMirror<SurfacePhysicsParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
