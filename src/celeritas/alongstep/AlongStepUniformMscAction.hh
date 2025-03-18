//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepUniformMscAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/field/UniformFieldParams.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/inp/Field.hh"

namespace celeritas
{
class UrbanMscParams;
class FluctuationParams;
class PhysicsParams;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Along-step kernel with optional MSC and uniform magnetic field.
 */
class AlongStepUniformMscAction final : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Input = inp::UniformField;
    using SPConstFluctuations = std::shared_ptr<FluctuationParams const>;
    using SPConstMsc = std::shared_ptr<UrbanMscParams const>;
    using SPConstFieldParams = std::shared_ptr<UniformFieldParams const>;
    //!@}

  public:
    // Construct the along-step action from input parameters
    static std::shared_ptr<AlongStepUniformMscAction>
    from_params(ActionId id,
                GeoParams const& geometry,
                MaterialParams const& materials,
                ParticleParams const& particles,
                Input const& field_input,
                SPConstMsc msc,
                bool eloss_fluctuation);

    // Construct with next action ID, optional MSC, magnetic field
    AlongStepUniformMscAction(ActionId id,
                              GeoParams const& geometry,
                              Input const& field_input,
                              SPConstFluctuations fluct,
                              SPConstMsc msc);

    // Default destructor
    ~AlongStepUniformMscAction() final;

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the interaction kernel
    std::string_view label() const final { return "along-step-uniform-msc"; }

    //! Short description of the action
    std::string_view description() const final
    {
        return "apply along-step in a uniform field with Urban MSC";
    }

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::along; }

    //// ACCESSORS ////

    //! Whether energy flucutation is in use
    bool has_fluct() const { return static_cast<bool>(fluct_); }

    //! Whether MSC is in use
    bool has_msc() const { return static_cast<bool>(msc_); }

  private:
    ActionId id_;
    SPConstFieldParams field_;
    SPConstFluctuations fluct_;
    SPConstMsc msc_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
