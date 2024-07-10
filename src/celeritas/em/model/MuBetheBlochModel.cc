//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MuBetheBlochModel.cc
//---------------------------------------------------------------------------//
#include "MuBetheBlochModel.hh"

#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuBetheBlochData.hh"
#include "celeritas/em/executor/MuBetheBlochExecutor.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
MuBetheBlochModel::MuBetheBlochModel(ActionId id,
                                     ParticleParams const& particles)
    : ConcreteAction(id,
                     "ioni-mu-bethe-bloch",
                     "interact by muon ionization (Bethe-Bloch)")
{
    CELER_EXPECT(id);
    data_.electron = particles.find(pdg::electron());
    data_.mu_minus = particles.find(pdg::mu_minus());
    data_.mu_plus = particles.find(pdg::mu_plus());

    CELER_VALIDATE(data_.electron && data_.mu_minus && data_.mu_plus,
                   << "missing electron and/or muon particles (required for "
                   << this->description() << ")");

    data_.electron_mass = particles.get(data_.electron).mass();

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MuBetheBlochModel::applicability() const -> SetApplicability
{
    Applicability mu_minus_applic;
    mu_minus_applic.particle = data_.mu_minus;
    mu_minus_applic.lower = detail::mu_bethe_bloch_lower_limit();
    mu_minus_applic.upper = detail::high_energy_limit();

    Applicability mu_plus_applic = mu_minus_applic;
    mu_plus_applic.particle = data_.mu_plus;

    return {mu_minus_applic, mu_plus_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto MuBetheBlochModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Aside from the production cut, the discrete interaction is material
    // independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void MuBetheBlochModel::execute(CoreParams const& params,
                                CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{MuBetheBlochExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void MuBetheBlochModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
