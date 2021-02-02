//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GammaAnnihilationProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include "physics/base/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Annihiliation process for photons.
 */
class GammaAnnihilationProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    //!@}

  public:
    // Construct from particle data
    explicit GammaAnnihilationProcess(SPConstParticles particles);

    // Construct the models associated with this process
    VecModel build_models(ModelIdGenerator next_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles particles_;
    ParticleId       positron_id_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
