//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/EPlusAnnihilationProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Types.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
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
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

  public:
    // Construct from particle data
    explicit EPlusAnnihilationProcess(SPConstParticles particles,
                                      SPConstImported process_data);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Whether the integral method can be used to sample interaction length
    bool supports_integral_xs() const final { return true; }

    //! Whether the process applies when the particle is stopped
    bool applies_at_rest() const final { return applies_at_rest_; }

    // Name of the process
    std::string_view label() const final;

  private:
    SPConstParticles particles_;
    ParticleId positron_id_;
    bool applies_at_rest_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
