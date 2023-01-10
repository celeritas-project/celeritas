//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/MultipleScatteringProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Process.hh"

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
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

  public:
    // Construct with imported data
    MultipleScatteringProcess(SPConstParticles particles,
                              SPConstMaterials materials,
                              SPConstImported process_data);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability applicability) const final;

    //! Whether to use the integral method to sample interaction length
    bool use_integral_xs() const final { return false; }

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles particles_;
    SPConstMaterials materials_;
    ImportedProcessAdapter imported_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
