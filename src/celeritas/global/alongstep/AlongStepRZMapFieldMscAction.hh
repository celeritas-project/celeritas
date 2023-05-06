//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepRZMapFieldMscAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/field/RZMapFieldData.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
class UrbanMscParams;
class FluctuationParams;
class PhysicsParams;
class MaterialParams;
class ParticleParams;
struct RZMapFieldInput;

//---------------------------------------------------------------------------//
/*!
 * Along-step kernel with MSC, energy loss fluctuations, and a RZMapField.
 */
class AlongStepRZMapFieldMscAction final : public ExplicitActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFluctuations = std::shared_ptr<FluctuationParams const>;
    using SPConstMsc = std::shared_ptr<UrbanMscParams const>;
    using SPConstFieldParams = std::shared_ptr<RZMapFieldParams const>;
    //!@}

  public:
    static std::shared_ptr<AlongStepRZMapFieldMscAction>
    from_params(ActionId id,
                MaterialParams const& materials,
                ParticleParams const& particles,
                RZMapFieldInput const& field_input,
                SPConstMsc const& msc);

    // Construct with next action ID, optional MSC, magnetic field
    AlongStepRZMapFieldMscAction(ActionId id,
                                 SPConstFluctuations fluct,
                                 RZMapFieldInput const& input,
                                 SPConstMsc msc);

    // Launch kernel with host data
    void execute(CoreParams const&, StateHostRef&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, StateDeviceRef&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the interaction kernel
    std::string label() const final { return "along-step-rzmap-msc"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "along-step in a r-z map field with Urban MSC";
    }

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::along; }

    //// ACCESSORS ////

    //! Whether MSC is in use
    bool has_msc() const { return static_cast<bool>(msc_); }

    //! Field map data
    SPConstFieldParams const& field() const { return field_; }

  private:
    ActionId id_;
    SPConstFluctuations fluct_;
    SPConstMsc msc_;
    SPConstFieldParams field_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
inline void
AlongStepRZMapFieldMscAction::execute(CoreParams const&, StateDeviceRef&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
