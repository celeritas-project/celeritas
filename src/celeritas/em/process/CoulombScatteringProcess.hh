//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/CoulombScatteringProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/em/data/CoulombScatteringData.hh"
#include "celeritas/em/model/CoulombScatteringModel.hh"
#include "celeritas/io/ImportParameters.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Coulomb scattering process for electrons off of atoms.
 */
class CoulombScatteringProcess : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

    struct Options
    {
        //! Whether to use integral method to sample interaction length
        bool use_integral_xs{true};
    };

  public:
    //! Construct from Coulomb scattering data
    CoulombScatteringProcess(SPConstParticles particles,
                             SPConstMaterials materials,
                             SPConstImported process_data,
                             Options const& options);

    //! Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    //! Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Whether to use the integral method to sample interaction length
    bool use_integral_xs() const final;

    //! Whether the process applies when the particle is stopped
    bool applies_at_rest() const final { return imported_.applies_at_rest(); }

    // Name of the process
    std::string_view label() const final;

  private:
    SPConstParticles particles_;
    SPConstMaterials materials_;
    ImportedProcessAdapter imported_;
    Options options_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
