//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BetheHeitlerModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "celeritas/em/data/BetheHeitlerData.hh"

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
    BetheHeitlerModel(ActionId              id,
                      const ParticleParams& particles,
                      bool                  enable_lpm);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel on host
    void execute(CoreHostRef const&) const final;

    // Apply the interaction kernel on device
    void execute(CoreDeviceRef const&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "conv-bethe-heitler"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "Bethe-Heitler gamma conversion";
    }

  private:
    detail::BetheHeitlerData interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
