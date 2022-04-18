//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MultipleScatteringProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include "physics/base/ImportedProcessAdapter.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/material/MaterialParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Ionization process for electrons and positrons.
 */
class MultipleScatteringProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstMaterials = std::shared_ptr<const MaterialParams>;
    using SPConstImported  = std::shared_ptr<const ImportedProcesses>;
    //!@}

  public:
    // Construct with imported data
    MultipleScatteringProcess(SPConstParticles particles,
                              SPConstMaterials materials,
                              SPConstImported  process_data);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability applicability) const final;

    //! Type of process
    ProcessType type() const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles       particles_;
    SPConstMaterials       materials_;
    ImportedProcessAdapter imported_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
