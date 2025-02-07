//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/process/NeutronInelasticProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
struct ImportPhysicsVector;

//---------------------------------------------------------------------------//
/*!
 * Inelastic interaction process for neutrons.
 */
class NeutronInelasticProcess : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using ReadData = std::function<ImportPhysicsVector(AtomicNumber)>;
    //!@}

  public:
    // Construct from particle, material, and external cross section data
    NeutronInelasticProcess(SPConstParticles particles,
                            SPConstMaterials materials,
                            ReadData load_data);

    // Construct the models associated with this process
    VecModel build_models(ActionIdIter start_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    //! Whether to use the integral method to sample interaction length
    bool use_integral_xs() const final { return false; }

    //! Whether the process applies when the particle is stopped
    bool applies_at_rest() const final { return false; }

    // Name of the process
    std::string_view label() const final;

  private:
    SPConstParticles particles_;
    SPConstMaterials materials_;
    ReadData load_data_;
    ParticleId neutron_id_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
