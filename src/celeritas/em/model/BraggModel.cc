//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BraggModel.cc
//---------------------------------------------------------------------------//
#include "BraggModel.hh"

#include "celeritas/Quantities.hh"
#include "celeritas/em/data/BraggICRU73QOData.hh"
#include "celeritas/em/executor/BraggICRU73QOExecutor.hh"
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
 *
 * TODO: This model also applies to hadrons.
 */
BraggModel::BraggModel(ActionId id, ParticleParams const& particles)
    : ConcreteAction(id, "ioni-bragg", "interact by muon ionization (Bragg)")
{
    CELER_EXPECT(id);

    data_.inc_particle = particles.find(pdg::mu_plus());
    data_.electron = particles.find(pdg::electron());

    CELER_VALIDATE(data_.electron && data_.inc_particle,
                   << "missing electron and/or mu+ particles (required for "
                   << this->description() << ")");

    data_.electron_mass = particles.get(data_.electron).mass();
    data_.lowest_kin_energy = detail::bragg_lowest_kin_energy();

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto BraggModel::applicability() const -> SetApplicability
{
    Applicability applic;
    applic.particle = data_.inc_particle;
    applic.lower = zero_quantity();
    applic.upper = detail::mu_bethe_bloch_lower_limit();

    return {applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto BraggModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Aside from the production cut, the discrete interaction is material
    // independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void BraggModel::execute(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{BraggICRU73QOExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void BraggModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
