//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BetheHeitlerModel.cc
//---------------------------------------------------------------------------//
#include "BetheHeitlerModel.hh"

#include <utility>

#include "celeritas/Quantities.hh"
#include "celeritas/em/executor/BetheHeitlerExecutor.hh"  // IWYU pragma: associated
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/InteractionApplier.hh"  // IWYU pragma: associated
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
BetheHeitlerModel::BetheHeitlerModel(ActionId id,
                                     ParticleParams const& particles,
                                     SPConstImported data,
                                     bool enable_lpm)
    : StaticConcreteAction(id,
                           "conv-bethe-heitler",
                           "interact by Bethe-Heitler gamma conversion")
    , imported_(data,
                particles,
                ImportProcessClass::conversion,
                ImportModelClass::bethe_heitler_lpm,
                {pdg::gamma()})
{
    CELER_EXPECT(id);
    data_.ids.electron = particles.find(pdg::electron());
    data_.ids.positron = particles.find(pdg::positron());
    data_.ids.gamma = particles.find(pdg::gamma());
    data_.enable_lpm = enable_lpm;

    CELER_VALIDATE(data_.ids,
                   << "missing electron, positron and/or gamma particles "
                      "(required for "
                   << this->description() << ")");
    data_.electron_mass = particles.get(data_.ids.electron).mass();
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto BetheHeitlerModel::applicability() const -> SetApplicability
{
    using Energy = units::MevEnergy;

    Applicability photon_applic;
    photon_applic.particle = data_.ids.gamma;
    photon_applic.lower = Energy{2 * this->host_ref().electron_mass.value()};
    photon_applic.upper = Energy{1e8};

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto BetheHeitlerModel::micro_xs(Applicability applic) const -> XsTable
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void BetheHeitlerModel::step(CoreParams const& params,
                             CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{BetheHeitlerExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void BetheHeitlerModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
