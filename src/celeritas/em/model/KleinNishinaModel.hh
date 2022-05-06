//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/KleinNishinaModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "celeritas/em/data/KleinNishinaData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Klein-Nishina model interaction.
 */
class KleinNishinaModel final : public Model
{
  public:
    // Construct from model ID and other necessary data
    KleinNishinaModel(ActionId id, const ParticleParams& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    //! Apply the interaction kernel to host data
    void execute(CoreHostRef const&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreDeviceRef const&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "scat-klein-nishina"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "Klein-Nishina Compton scattering";
    }

  private:
    detail::KleinNishinaData interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
