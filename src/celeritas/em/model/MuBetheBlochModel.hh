//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MuBetheBlochModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/MuBetheBlochData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Bethe-Bloch muon ionization model interaction.
 */
class MuBetheBlochModel final : public Model
{
  public:
    // Construct from model ID and other necessary data
    MuBetheBlochModel(ActionId id, ParticleParams const& particles);

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
    std::string_view label() const final { return "ioni-mu-bethe-bloch"; }

    //! Short description of the post-step action
    std::string_view description() const final
    {
        return "interact by muon ionization (Bethe-Bloch)";
    }

    //!@{
    //! Access model data
    MuBetheBlochData const& host_ref() const { return data_; }
    MuBetheBlochData const& device_ref() const { return data_; }
    //!@}

  private:
    MuBetheBlochData data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
