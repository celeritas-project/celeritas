//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/channel/MuDecayChannel.cc
//---------------------------------------------------------------------------//
#include "MuDecayChannel.hh"

#include "corecel/Assert.hh"
#include "celeritas/Constants.hh"
#include "celeritas/decay/executor/MuDecayExecutor.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from action ID, particles, and input.
 */
MuDecayChannel::MuDecayChannel(
    ActionId id, ParticleParams const& particles, inp::MuonDecay const& input)
    : StaticConcreteAction(id, "decay-muon", "decay of muons to e-/e+")
{
    CELER_EXPECT(id);
    CELER_VALIDATE(input, << "no particles are enabled for muon decay");

    data_.mu_minus_id = particles.find(pdg::mu_minus());
    data_.mu_plus_id = particles.find(pdg::mu_plus());
    data_.electron_id = particles.find(pdg::electron());
    data_.positron_id = particles.find(pdg::positron());
    CELER_VALIDATE(data_.mu_minus_id && data_.mu_plus_id && data_.electron_id
                       && data_.positron_id,
                   << "missing muon and/or electron particles (required for "
                   << this->description() << ")");
    data_.muon_mass = particles.get(data_.mu_minus_id).mass();
    data_.electron_mass = particles.get(data_.electron_id).mass();

    for (PDGNumber pdg : input.parents)
    {
        CELER_VALIDATE(pdg == pdg::mu_minus() || pdg == pdg::mu_plus(),
                       << "invalid parent particle PDG " << pdg.get()
                       << " for muon decay (must be mu- or mu+)");
        ParticleId pid = particles.find(pdg);
        CELER_VALIDATE(particles.get(pid).decay_constant()
                           != constants::stable_decay_constant,
                       << "muon decay is enabled but particle PDG "
                       << pdg.get() << " is stable");
        parents_.push_back(pid);
    }

    CELER_ENSURE(data_ && !parents_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel on host.
 */
void MuDecayChannel::step(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{MuDecayExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void MuDecayChannel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
