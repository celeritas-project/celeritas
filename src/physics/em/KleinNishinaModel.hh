//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"
#include "physics/base/ParticleParams.hh"

#include "detail/KleinNishinaData.hh"

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
    void interact(const HostInteractRef&) const final;

    // Apply the interaction kernel to device data
    void interact(const DeviceInteractRef&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Klein-Nishina"; }

  private:
    detail::KleinNishinaData interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
