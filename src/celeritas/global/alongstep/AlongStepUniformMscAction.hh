//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepUniformMscAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
class UrbanMscParams;
class PhysicsParams;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Along-step kernel with optional MSC and uniform magnetic field.
 */
class AlongStepUniformMscAction final : public ExplicitActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstMsc = std::shared_ptr<UrbanMscParams const>;
    //!@}

  public:
    // Construct with next action ID, optional MSC, magnetic field
    AlongStepUniformMscAction(ActionId id,
                              UniformFieldParams const& field_params,
                              SPConstMsc msc);

    // Default destructor
    ~AlongStepUniformMscAction();

    // Launch kernel with host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the interaction kernel
    std::string label() const final { return "along-step-uniform-msc"; }

    //! Short description of the action
    std::string description() const final
    {
        return "apply along-step in a uniform field with Urban MSC";
    }

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::along; }

    //// ACCESSORS ////

    //! Whether MSC is in use
    bool has_msc() const { return static_cast<bool>(msc_); }

    //! Field strength
    Real3 const& field() const { return field_params_.field; }

  private:
    ActionId id_;
    SPConstMsc msc_;
    UniformFieldParams field_params_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
