//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Model.hh"

#include "physics/base/ParticleParams.hh"
#include "detail/EPlusGG.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch two-gamma positron annihiliation.
 */
class EPlusGGModel final : public Model
{
  public:
    // Construct from model ID and other necessary data
    EPlusGGModel(ModelId id, const ParticleParams& particles);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel
    void interact(const ModelInteractPointers&) const final;

    // ID of the model
    ModelId model_id() const final;

    //! Name of the model, for user interaction
    std::string label() const final { return "Positron annihilation (2g)"; }

    // Access data on device
    detail::EPlusGGPointers device_pointers() const { return interface_; }

  private:
    detail::EPlusGGPointers interface_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
