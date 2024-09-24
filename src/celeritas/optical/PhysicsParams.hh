//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"

#include "PhysicsData.hh"

namespace celeritas
{
class ActionRegistry;

namespace optical
{
struct ModelBuilder;
class Model;

//---------------------------------------------------------------------------//

struct PhysicsParamsOptions
{
};

class PhysicsParams : public ParamsDataInterface<PhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstModelBuilder = std::shared_ptr<ModelBuilder const>;
    using SPConstModel = std::shared_ptr<Model const>;

    using ActionIdRange = Range<ActionId>;
    using Options = PhysicsParamsOptions;
    //!@}

    //! Optical physics parameter construction arguments
    struct Input
    {
        std::vector<SPConstModelBuilder> model_builders;

        ActionRegistry* action_registry = nullptr;
        Options options;
    };

  public:
    // Construct with models and helper classes
    explicit PhysicsParams(Input);

    //// HOST ACCESSORS ////

    // Number of optical models
    inline ModelId::size_type num_models() const { return models_.size(); }

    // Get a model
    inline SPConstModel const& model(ModelId mid) const
    {
        CELER_EXPECT(mid && mid < num_models());
        return models_[mid.get()];
    }

    // Get the action IDs for all models
    inline ActionIdRange model_actions() const
    {
        auto offset = host_ref().scalars.model_to_action;
        return {ActionId{offset}, ActionId{offset + this->num_models()}};
    }

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    using SPAction = std::shared_ptr<ConcreteAction>;
    using HostValue = HostVal<PhysicsParamsData>;
    using VecModels = std::vector<SPConstModel>;

    SPAction discrete_action_;

    VecModels models_;

    CollectionMirror<PhysicsParamsData> data_;

    VecModels
    build_models(std::vector<SPConstModelBuilder> const& model_builders,
                 ActionRegistry& action_reg) const;
    void build_options(Options const& options, HostValue& host_data) const;
    void build_mfps(HostValue& host_data) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
