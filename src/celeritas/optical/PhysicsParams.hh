//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "ModelBuilder.hh"
#include "PhysicsData.hh"
#include "action/ActionInterface.hh"

namespace celeritas
{
class ActionRegistry;
class ConcreteAction;

namespace optical
{
class MaterialParams;

//---------------------------------------------------------------------------//
class PhysicsParams final : public ParamsDataInterface<PhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstModel = std::shared_ptr<Model const>;
    using SPConstModelBuilder = std::shared_ptr<ModelBuilder const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;

    using VecModels = std::vector<SPConstModel>;
    using VecModelBuilders = std::vector<SPConstModelBuilder>;

    using ActionIdRange = Range<ActionId>;
    //!@}

    struct Input
    {
        VecModelBuilders model_builders;
        SPConstMaterials materials;
        ActionRegistry* action_registry = nullptr;
    };

  public:
    explicit PhysicsParams(Input input);

    //! Access optical physics data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical physics data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    using SPAction = std::shared_ptr<ConcreteAction const>;
    using HostValue = HostVal<PhysicsParamsData>;

    // Actions
    SPAction discrete_select_;
    VecModels models_;

    // Host/device storage
    CollectionMirror<PhysicsParamsData> data_;

    //!@{
    //! \name Data construction helper functions
    VecModels build_models(VecModelBuilders const& model_builders,
                           ActionRegistry& action_reg) const;
    void build_mfps(MaterialParams const& mats, HostValue& data) const;
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
