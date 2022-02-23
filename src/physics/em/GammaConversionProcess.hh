//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GammaConversionProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include <memory>
#include "physics/base/ImportedProcessAdapter.hh"
#include "physics/base/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Conversion of gammas to electrons and positrons.
 */
class GammaConversionProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstImported  = std::shared_ptr<const ImportedProcesses>;
    //!@}

  public:
    // Construct from particle data
    GammaConversionProcess(SPConstParticles particles,
                           SPConstImported  process_data);

    // Construct the models associated with this process
    VecModel build_models(ModelIdGenerator next_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability applic) const final;

    //! Type of process
    ProcessType type() const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles       particles_;
    ImportedProcessAdapter imported_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
