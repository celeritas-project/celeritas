//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MuBremsstrahlungModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/MuBremsstrahlungData.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Muon Bremsstrahlung model interaction.
 */
class MuBremsstrahlungModel final : public Model
{
  public:
    // Construct from model ID and other necessary data
    MuBremsstrahlungModel(ActionId id, const ParticleParams& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel on host
    void execute(CoreHostRef const&) const final;

    // Apply the interaction kernel on device
    void execute(CoreDeviceRef const&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "brems-muon"; }

    //! Name of the model, for user interaction
    std::string description() const final { return "Muon bremsstrahlung"; }

  private:
    detail::MuBremsstrahlungData interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
