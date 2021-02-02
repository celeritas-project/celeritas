//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Process.hh"

#include "io/ImportPhysicsTable.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/em/LivermorePEParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Photoelectric effect process for gammas.
 */
class PhotoelectricProcess : public Process
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstData      = std::shared_ptr<const LivermorePEParams>;
    //!@}

  public:
    // Construct from Livermore photoelectric data
    PhotoelectricProcess(SPConstParticles particles, SPConstData data);

    // Construct the models associated with this process
    VecModel build_models(ModelIdGenerator next_id) const final;

    // Get the interaction cross sections for the given energy range
    StepLimitBuilders step_limits(Applicability range) const final;

    // Name of the process
    std::string label() const final;

  private:
    SPConstParticles particles_;
    SPConstData      data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
