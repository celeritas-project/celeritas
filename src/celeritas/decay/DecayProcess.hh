//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/DecayProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Types.hh"
#include "celeritas/decay/channel/DecayChannel.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Process.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Process for decay.
 */
class DecayProcess final : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstChannel = std::shared_ptr<DecayChannel const>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using VecChannel = std::vector<SPConstChannel>;
    using DecayTable = inp::DecayPhysics::DecayTable;
    //!@}

  public:
    // Construct from particles and input data
    DecayProcess(SPConstParticles, inp::DecayPhysics const&);

    // Construct the decay channels
    VecChannel build_channels(ActionIdIter start_id) const;

    // Get the decay table for a particle
    DecayTable decay_table(ParticleId) const;

    // Whether the decay process applies to the particle
    bool is_applicable(ParticleId) const;

    //!@{
    //! \name Process interface

    //! Whether the process applies when the particle is stopped
    bool applies_at_rest() const final { return true; }
    //! Name of the process
    std::string_view label() const final { return "Decay"; }
    //!@}

  private:
    SPConstParticles particles_;
    inp::DecayPhysics const& input_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
