//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/optical/action/ActionInterface.hh"

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
    using SPConstInteractionModel = std::shared_ptr<SurfaceModel const>;

    using VecRoughnessModels = std::vector<SPConstRoughnessModel>;
    using VecReflectivityModels = std::vector<SPConstReflectivityModel>;
    using VecInteractionModels = std::vector<SPConstModel>;

    using ActionIdRange = Range<ActionId>;
    //!@}

    struct SurfaceInput
    {
        SurfaceRoughnessModelId roughness_model;
        SurfaceReflectivityModelId reflectivity_model;
        SurfaceModelId interaction_model;
    };

    struct Input
    {
        std::vector<SurfaceRoughnessModelBuilder> roughness_model_builders;
        std::vector<SurfaceReflectivityModelBuilder> reflectivity_model_builders;
        std::vector<SurfaceModelBuilder> interaction_model_builders;

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

  private:
    // Actions
    VecRoughnessModels roughness_models_;
    VecReflectivityModels reflectivity_models_;
    VecInteractionModels interaction_models_;

    // Host/device storage
    CollectionMirror<SurfacePhysicsParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
