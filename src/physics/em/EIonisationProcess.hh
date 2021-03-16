//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EIonisationProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include "physics/base/ParticleParams.hh"
#include "io/ImportPhysicsTable.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Ionisation process for electrons and positrons.
 */
class EIonisationProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    //!@}

    struct Input
    {
        SPConstParticles   particles;
        ImportPhysicsTable lambda;
        ImportPhysicsTable dedx;
        ImportPhysicsTable range;
    };

  public:
    // Construct with lambda table
    inline EIonisationProcess(const Input& input);

    // Construct the models associated with this process
    VecModel build_models(ModelIdGenerator next_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles   particles_;
    ImportPhysicsTable lambda_table_;
    ImportPhysicsTable dedx_table_;
    ImportPhysicsTable range_table_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
