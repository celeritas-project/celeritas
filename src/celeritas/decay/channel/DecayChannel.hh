//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/channel/DecayChannel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Abstract base class for a decay channel.
 *
 * A decay channel is the decay analog of a \c Model : it is the step action
 * that samples the decay products of a particle that the \c DecayProcess
 * selected to decay. Each channel knows the particle types it applies to and
 * what they decay into.
 */
class DecayChannel : public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecParticle = std::vector<ParticleId>;
    //!@}

  public:
    //! Particle types that decay through this channel
    virtual VecParticle const& parents() const = 0;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::post; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
