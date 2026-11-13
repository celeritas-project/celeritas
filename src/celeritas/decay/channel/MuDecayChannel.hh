//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/channel/MuDecayChannel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/decay/data/MuDecayData.hh"
#include "celeritas/inp/DecayPhysics.hh"

#include "DecayChannel.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Decay of muons and anti-muons to an electron or positron.
 *
 * This launches \c MuDecayInteractor for tracks that decay through this
 * channel.
 */
class MuDecayChannel final : public DecayChannel, public StaticConcreteAction
{
  public:
    // Construct from action ID, particles, and input
    MuDecayChannel(ActionId, ParticleParams const&, inp::MuonDecay const&);

    //! Particle types that decay through this channel
    VecParticle const& parents() const final { return parents_; }

    //!@{
    //! \name StepAction interface

    // Apply the interaction kernel on host
    void step(CoreParams const&, CoreStateHost&) const final;
    // Apply the interaction kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

    //!@{
    //! Access data
    MuDecayData const& host_ref() const { return data_; }
    MuDecayData const& device_ref() const { return data_; }
    //!@}

  private:
    MuDecayData data_;
    VecParticle parents_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
