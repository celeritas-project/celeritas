//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusAnnihilationProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include <memory>
#include "physics/base/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Annihiliation process for positrons.
 */
class EPlusAnnihilationProcess final : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    //!@}

  public:
    // Construct from particle data
    explicit EPlusAnnihilationProcess(SPConstParticles particles);

    // Construct the models associated with this process
    VecModel build_models(ModelIdGenerator next_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Type of process
    ProcessType type() const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles particles_;
    ParticleId       positron_id_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
