//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/CoulombScatteringProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/em/data/WentzelData.hh"
#include "celeritas/io/ImportParameters.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{

namespace detail
{
//! Coulomb Scattering configuration options
struct CoulombScatteringOptions
{
    //! Nuclear form factor model
    NuclearFormFactorType form_factor_model{NuclearFormFactorType::exponential};

    //! User defined screening factor
    real_type screening_factor{1};
};
}  // namespace detail

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
    using Options = detail::CoulombScatteringOptions;
    //!@}

  public:
    //! Construct from Wentzel scattering data
    CoulombScatteringProcess(SPConstParticles particles,
                             SPConstMaterials materials,
                             SPConstImported process_data,
                             Options const& options);

    //! Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    //! Get the interaction cross sections fro the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Whether to use the integral method to sample interaction length
    bool use_integral_xs() const final { return false; }

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles particles_;
    SPConstMaterials materials_;
    ImportedProcessAdapter imported_;
    Options options_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
