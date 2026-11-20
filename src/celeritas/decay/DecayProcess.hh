//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/DecayProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/inp/DecayPhysics.hh"
#include "celeritas/phys/Process.hh"

#include "channel/DecayChannel.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Spontaneous decay of unstable particles.
 *
 * This single process manages all supported decay channels, each of which is
 * analogous to a model of an \c InteractionProcess : the channel is the step
 * action that samples the decay products. Unlike an interaction with matter,
 * the decay rate depends only on the particle: in flight, the decay "cross
 * section" is the inverse of the mean decay length
 * \f[
   \Sigma_\mathrm{decay} = \frac{1}{\beta \gamma c \tau}
                         = \frac{\lambda m}{c p},
 * \f]
 * where \f$ \lambda = 1/\tau \f$ is the particle's decay constant. As in
 * Geant4, it is evaluated at the momentum at the start of the step.
 *
 * Decay also applies at rest, where it is selected only if the particle has
 * no other at-rest process. This mirrors Geant4, where at-rest processes such
 * as \f$\mu^-\f$ capture occur immediately and always preempt decay.
 *
 * Each particle type may currently decay through at most one channel.
 */
class DecayProcess final : public Process
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstChannel = std::shared_ptr<DecayChannel const>;
    using VecChannel = std::vector<SPConstChannel>;
    //!@}

  public:
    // Construct from particles and input
    DecayProcess(SPConstParticles particles, inp::DecayPhysics input);

    // Construct the decay channels
    VecChannel build_channels(ActionIdIter start_id) const;

    //!@{
    //! \name Process interface

    //! Whether the process applies when the particle is stopped
    bool applies_at_rest() const final { return true; }
    //! Name of the process
    std::string_view label() const final { return "Decay"; }
    //!@}

  private:
    SPConstParticles particles_;
    inp::DecayPhysics input_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
