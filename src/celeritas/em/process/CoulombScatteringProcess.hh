//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/CoulombScatteringProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/io/ImportParameters.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Coulomb scattering process for electrons off of atoms
 */
class CoulombScatteringProcess : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    using SPConstEmParameters = std::shared_ptr<ImportEmParameters const>;
    //!@}

  public:
    CoulombScatteringProcess(SPConstParticles particles,
                             SPConstMaterials materials,
                             SPConstImported process_data,
                             SPConstEmParameters em_params);

    VecModel build_models(ActionIdIter start_id) const final;
    StepLimitBuilders step_limits(Applicability range) const final;
    bool use_integral_xs() const final { return false; }

    std::string label() const final;

  private:
    SPConstParticles particles_;
    SPConstMaterials materials_;
    ImportedProcessAdapter imported_;
    SPConstEmParameters em_params_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
