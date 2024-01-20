//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MollerBhabhaModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/MollerBhabhaData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Moller-Bhabha model interaction.
 */
class MollerBhabhaModel final : public Model
{
  public:
    // Construct from model ID and other necessary data
    MollerBhabhaModel(ActionId id, ParticleParams const& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel on device
    void execute(CoreParams const&, CoreStateDevice&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "ioni-moller-bhabha"; }

    //! Short description of the post-step action
    std::string description() const final
    {
        return "interact by Moller+Bhabha ionization";
    }

    //!@{
    //! Access model data
    MollerBhabhaData const& host_ref() const { return data_; }
    MollerBhabhaData const& device_ref() const { return data_; }
    //!@}

  private:
    MollerBhabhaData data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
