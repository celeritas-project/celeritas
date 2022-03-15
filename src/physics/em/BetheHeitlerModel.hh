//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"
#include "physics/base/ParticleParams.hh"

#include "detail/BetheHeitlerData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Bethe-Heitler model interaction.
 */
class BetheHeitlerModel final : public Model
{
  public:
    // Construct from model ID and other necessary data
    BetheHeitlerModel(ModelId               id,
                      const ParticleParams& particles,
                      bool                  enable_lpm = true);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel on host
    void interact(const HostInteractRef&) const final;

    // Apply the interaction kernel on device
    void interact(const DeviceInteractRef&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Bethe-Heitler"; }

  private:
    detail::BetheHeitlerData interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
