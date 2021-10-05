//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MuBremsstrahlungModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"
#include "physics/base/ParticleParams.hh"
#include "detail/MuBremsstrahlungInterface.hh"

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
    MuBremsstrahlungModel(ModelId id, const ParticleParams& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel on host
    void interact(const HostInteractRefs&) const final;

    // Apply the interaction kernel on device
    void interact(const DeviceInteractRefs&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Muon Bremsstrahlung"; }

  private:
    detail::MuBremsstrahlungPointers interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
