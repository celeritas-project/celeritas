//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ComptonProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include "physics/base/ParticleParams.hh"
#include "io/ImportPhysicsTable.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Compton scattering process for gammas.
 */
class ComptonProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    //!@}

  public:
    // Construct from "lambda" and "lambda_prim" tables
    ComptonProcess(SPConstParticles   particles,
                   ImportPhysicsTable xs_lo,
                   ImportPhysicsTable xs_hi);

    // Construct the models associated with this process
    VecModel build_models(ModelIdGenerator next_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles   particles_;
    ImportPhysicsTable xs_lo_;
    ImportPhysicsTable xs_hi_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
