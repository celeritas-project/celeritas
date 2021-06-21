//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheBlochModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"
#include "physics/base/ParticleParams.hh"
#include "detail/BetheBloch.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Bethe-Bloch model interaction.
 */
class BetheBlochModel final : public Model
{
  public:
    // Construct from model ID and other necessary data
    BetheBlochModel(ModelId id, const ParticleParams& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel
    void interact(const DeviceInteractRefs&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Bethe-Bloch"; }

  private:
    detail::BetheBlochInteractorPointers interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

