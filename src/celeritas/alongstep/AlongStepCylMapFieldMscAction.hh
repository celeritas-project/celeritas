//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepCylMapFieldMscAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/field/CylMapFieldData.hh"
#include "celeritas/field/CylMapFieldParams.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
class UrbanMscParams;
class FluctuationParams;
class PhysicsParams;
class MaterialParams;
class ParticleParams;
struct CylMapFieldInput;

//---------------------------------------------------------------------------//
/*!
 * Along-step kernel with MSC, energy loss fluctuations, and a CylMapField.
 */
class AlongStepCylMapFieldMscAction final : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFluctuations = std::shared_ptr<FluctuationParams const>;
    using SPConstMsc = std::shared_ptr<UrbanMscParams const>;
    using SPConstFieldParams = std::shared_ptr<CylMapFieldParams const>;
    //!@}

  public:
    static std::shared_ptr<AlongStepCylMapFieldMscAction>
    from_params(ActionId id,
                MaterialParams const& materials,
                ParticleParams const& particles,
                CylMapFieldInput const& field_input,
                SPConstMsc const& msc,
                bool eloss_fluctuation);

    // Construct with next action ID and physics properties
    AlongStepCylMapFieldMscAction(ActionId id,
                                  CylMapFieldInput const& input,
                                  SPConstFluctuations fluct,
                                  SPConstMsc msc);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the interaction kernel
    std::string_view label() const final { return "along-step-cylmap-msc"; }

    //! Short description of the action
    std::string_view description() const final
    {
        return "apply along-step in a R-Phi-Z map field with Urban MSC";
    }

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::along; }

    //// ACCESSORS ////

    //! Whether energy flucutation is in use
    bool has_fluct() const { return static_cast<bool>(fluct_); }

    //! Whether MSC is in use
    bool has_msc() const { return static_cast<bool>(msc_); }

    //! Field map data
    SPConstFieldParams const& field() const { return field_; }

  private:
    ActionId id_;
    SPConstFieldParams field_;
    SPConstFluctuations fluct_;
    SPConstMsc msc_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
