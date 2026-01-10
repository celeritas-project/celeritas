//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "Model.hh"
#include "PhysicsData.hh"
#include "action/ActionInterface.hh"

namespace celeritas
{
class ActionRegistry;
struct ImportData;
class MaterialParams;

namespace optical
{
class MaterialParams;

//---------------------------------------------------------------------------//
class PhysicsParams final : public ParamsDataInterface<PhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;
    using SPConstModel = std::shared_ptr<Model const>;
    using SPConstCoreMaterials
        = std::shared_ptr<celeritas::MaterialParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;

    using VecModels = std::vector<SPConstModel>;
    using VecModelBuilders = std::vector<Model::ModelBuilder>;

    using ActionIdRange = Range<ActionId>;
    //!@}

    struct Input
    {
        VecModelBuilders model_builders;
        SPConstMaterials materials;
        ActionRegistry* action_registry = nullptr;
    };

  public:
    // Construct with imported data, material params, and action registry
    static std::shared_ptr<PhysicsParams> from_import(ImportData const&,
                                                      SPConstCoreMaterials,
                                                      SPConstMaterials,
                                                      SPActionRegistry);

    // Construct from models
    explicit PhysicsParams(Input input);

    //! Number of optical models
    inline ModelId::size_type num_models() const { return models_.size(); }

    // Get an optical model
    inline SPConstModel model(ModelId mid) const;

    // Get the action IDs for all models
    inline ActionIdRange model_actions() const;

    //! Access optical physics data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical physics data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    using SPAction = std::shared_ptr<StaticConcreteAction const>;
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
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get an optical model associated with the given model identifier.
 */
auto PhysicsParams::model(ModelId mid) const -> SPConstModel
{
    CELER_EXPECT(mid < this->num_models());
    return models_[mid.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the action identifiers for all optical models.
 */
auto PhysicsParams::model_actions() const -> ActionIdRange
{
    auto offset = host_ref().scalars.first_model_action;
    return {offset, offset + this->num_models()};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
