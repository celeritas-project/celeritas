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
    using SPConstModel = std::shared_ptr<SurfaceModel const>;
    using SPConstModelBuilder = std::shared_ptr<SurfaceModelBuilder const>;

    using VecModels = std::vector<SPConstModel>;
    using VecModelBuilders = std::vector<SPConstModelBuilder>;

    using ActionIdRange = Range<ActionId>;
    //!@}

    struct Input
    {
        VecModelBuilders model_builders;
        ActionRegistry* action_registry = nullptr;
    };

  public:
    // Construct from models
    explicit SurfacePhysicsParams(Input input);

    //! Number of surface models
    inline SurfaceModelId::size_type num_models() const
    {
        return models_.size();
    }

    // Get a surface model
    inline SPConstModel model(SurfaceModelId mid) const;

    // Get the boundary action for surface physics
    inline ActionId boundary_action() const;

    //! Access surface physics data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access surface physics data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Actions
    VecModels models_;

    // Host/device storage
    CollectionMirror<SurfacePhysicsParamsData> data_;

    //!@{
    //! \name Data construction helper functions
    VecModels build_models(VecModelBuilders const&, ActionRegistry&) const;
    //!@}
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get an optical surface model associated with the given model identifier.
 */
auto SurfacePhysicsParams::model(SurfaceModelId mid) const -> SPConstModel
{
    CELER_EXPECT(mid < this->num_models());
    return models_[mid.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the boundary action for surface physics.
 */
ActionId SurfacePhysicsParams::boundary_action() const
{
    // TODO: temporary placeholder for getting the initial boundary action
    CELER_EXPECT(models_.size() >= 1);
    return models_.front()->action_id();
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
