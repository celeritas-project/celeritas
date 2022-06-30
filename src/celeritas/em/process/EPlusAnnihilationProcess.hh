//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/EPlusAnnihilationProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Process.hh"

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

    // Options for electron-positron annihilation
    // TODO: update options based on ImportData
    struct Options
    {
        bool use_integral_xs{true}; //!> Use integral method for sampling
                                    //! discrete interaction length
    };

  public:
    // Construct from particle data
    explicit EPlusAnnihilationProcess(SPConstParticles particles,
                                      Options          options);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Whether to use the integral method to sample interaction length
    bool use_integral_xs() const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles particles_;
    ParticleId       positron_id_;
    Options          options_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
