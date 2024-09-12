//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/model/MuDecayModel.cc
//---------------------------------------------------------------------------//
#include "MuDecayModel.hh"

#include "celeritas/decay/executor/MuDecayExecutor.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
MuDecayModel::MuDecayModel(ActionId id, ParticleParams const& particles)
    : ConcreteAction(id, "mu-decay", "interact by muon decay")
{
    CELER_EXPECT(id);
    data_.ids.mu_minus = particles.find(pdg::mu_minus());
    data_.ids.mu_plus = particles.find(pdg::mu_plus());
    data_.ids.electron = particles.find(pdg::electron());
    data_.ids.positron = particles.find(pdg::positron());

    CELER_VALIDATE(data_.ids.mu_minus && data_.ids.mu_plus
                       && data_.ids.electron && data_.ids.positron,
                   << "missing muon, anti-muon, electron, and/or positron "
                      "particles (required for "
                   << this->description() << ")");

    data_.muon_mass = value_as<MuDecayData::Mass>(
        particles.get(data_.ids.mu_minus).mass());
    data_.electron_mass = value_as<MuDecayData::Mass>(
        particles.get(data_.ids.electron).mass());

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MuDecayModel::applicability() const -> SetApplicability
{
    Applicability mu_minus_applic;
    mu_minus_applic.particle = data_.ids.mu_minus;
    mu_minus_applic.lower = zero_quantity();
    mu_minus_applic.upper = max_quantity();

    auto mu_plus_applic = mu_minus_applic;
    mu_plus_applic.particle = data_.ids.mu_plus;

    return {mu_minus_applic, mu_plus_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto MuDecayModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Discrete interaction is material independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void MuDecayModel::step(CoreParams const& params, CoreStateHost& state) const
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
void MuDecayModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
