//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/CoulombScatteringModel.cc
//---------------------------------------------------------------------------//
#include "CoulombScatteringModel.hh"

#include <utility>

#include "celeritas_config.h"
#include "celeritas/Constants.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/data/CoulombScatteringData.hh"
#include "celeritas/em/executor/CoulombScatteringExecutor.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportParameters.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and shared data.
 */
CoulombScatteringModel::CoulombScatteringModel(ActionId id,
                                               ParticleParams const& particles,
                                               SPConstImported data)
    : imported_(data,
                particles,
                ImportProcessClass::coulomb_scat,
                ImportModelClass::e_coulomb_scattering,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(id);

    data_.action = id;
    data_.ids.electron = particles.find(pdg::electron());
    data_.ids.positron = particles.find(pdg::positron());

    CELER_VALIDATE(data_.ids,
                   << "missing electron and/or positron particles (required "
                      "for "
                   << this->description() << ")");

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto CoulombScatteringModel::applicability() const -> SetApplicability
{
    Applicability electron_applic;
    electron_applic.particle = this->host_ref().ids.electron;
    // TODO: Set the lower energy limit equal to the MSC energy limit when
    // combined single and multiple Coulomb scattering is supported and enabled
    electron_applic.lower = zero_quantity();
    electron_applic.upper = detail::high_energy_limit();

    Applicability positron_applic = electron_applic;
    positron_applic.particle = this->host_ref().ids.positron;

    return {electron_applic, positron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto CoulombScatteringModel::micro_xs(Applicability applic) const
    -> MicroXsBuilders
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void CoulombScatteringModel::execute(CoreParams const& params,
                                     CoreStateHost& state) const
{
    CELER_EXPECT(params.wentzel());

    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{CoulombScatteringExecutor{
            this->host_ref(), params.wentzel()->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void CoulombScatteringModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//!@}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for this model.
 */
ActionId CoulombScatteringModel::action_id() const
{
    return this->host_ref().action;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
