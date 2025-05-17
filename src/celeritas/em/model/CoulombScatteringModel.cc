//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/CoulombScatteringModel.cc
//---------------------------------------------------------------------------//
#include "CoulombScatteringModel.hh"

#include <utility>

#include "corecel/Config.hh"

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
#include "celeritas/mat/MaterialParams.hh"
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
    : StaticConcreteAction(
          id, "coulomb-wentzel", "interact by Coulomb scattering (Wentzel)")
    , imported_(data,
                particles,
                ImportProcessClass::coulomb_scat,
                ImportModelClass::e_coulomb_scattering,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(id);

    data_.ids.electron = particles.find(pdg::electron());
    data_.ids.positron = particles.find(pdg::positron());

    CELER_VALIDATE(
        data_.ids,
        << R"(missing electron and/or positron particles (required for )"
        << this->description() << ")");

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto CoulombScatteringModel::applicability() const -> SetApplicability
{
    return imported_.applicability();
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto CoulombScatteringModel::micro_xs(Applicability applic) const -> XsTable
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void CoulombScatteringModel::step(CoreParams const& params,
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
void CoulombScatteringModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
