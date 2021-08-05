//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"
#include "physics/base/ParticleParams.hh"
#include "detail/KleinNishina.hh"

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
    KleinNishinaModel(ModelId id, const ParticleParams& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    //! Apply the interaction kernel to host data
    void interact(const HostInteractRefs&) const final;

    // Apply the interaction kernel to device data
    void interact(const DeviceInteractRefs&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Klein-Nishina"; }

  private:
    detail::KleinNishinaPointers interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
