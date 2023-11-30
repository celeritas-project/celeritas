//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/EPlusGGModel.cc
//---------------------------------------------------------------------------//
#include "EPlusGGModel.hh"

#include "celeritas/Quantities.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/em/executor/EPlusGGExecutor.hh"  // IWYU pragma: associated
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"  // IWYU pragma: associated
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
EPlusGGModel::EPlusGGModel(ActionId id, ParticleParams const& particles)
{
    CELER_EXPECT(id);
    data_.ids.action = id;
    data_.ids.positron = particles.find(pdg::positron());
    data_.ids.gamma = particles.find(pdg::gamma());

    CELER_VALIDATE(data_.ids.positron && data_.ids.gamma,
                   << "missing positron and/or gamma particles (required for "
                   << this->description() << ")");
    data_.electron_mass = particles.get(data_.ids.positron).mass();
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto EPlusGGModel::applicability() const -> SetApplicability
{
    Applicability applic;
    applic.particle = data_.ids.positron;
    applic.lower = zero_quantity();  // Valid at rest
    applic.upper = units::MevEnergy{1e8};  // 100 TeV

    return {applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto EPlusGGModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Discrete interaction is material independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void EPlusGGModel::execute(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{EPlusGGExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void EPlusGGModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId EPlusGGModel::action_id() const
{
    return data_.ids.action;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
