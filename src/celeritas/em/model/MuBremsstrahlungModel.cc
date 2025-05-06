//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MuBremsstrahlungModel.cc
//---------------------------------------------------------------------------//
#include "MuBremsstrahlungModel.hh"

#include <utility>

#include "celeritas/Quantities.hh"
#include "celeritas/em/executor/MuBremsstrahlungExecutor.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
MuBremsstrahlungModel::MuBremsstrahlungModel(ActionId id,
                                             ParticleParams const& particles,
                                             SPConstImported data)
    : StaticConcreteAction(
          id, "brems-muon", "interact by bremsstrahlung (muon)")
    , imported_(data,
                particles,
                ImportProcessClass::mu_brems,
                ImportModelClass::mu_brems,
                {pdg::mu_minus(), pdg::mu_plus()})
{
    CELER_EXPECT(id);
    data_.gamma = particles.find(pdg::gamma());
    data_.mu_minus = particles.find(pdg::mu_minus());
    data_.mu_plus = particles.find(pdg::mu_plus());

    CELER_VALIDATE(data_.gamma && data_.mu_minus && data_.mu_plus,
                   << "missing muon and/or gamma particles (required for "
                   << this->description() << ")");

    data_.electron_mass = particles.get(particles.find(pdg::electron())).mass();
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MuBremsstrahlungModel::applicability() const -> SetApplicability
{
    Applicability mu_minus_applic, mu_plus_applic;

    mu_minus_applic.particle = data_.mu_minus;
    mu_minus_applic.lower = zero_quantity();
    mu_minus_applic.upper = detail::high_energy_limit();

    mu_plus_applic.particle = data_.mu_plus;
    mu_plus_applic.lower = mu_minus_applic.lower;
    mu_plus_applic.upper = mu_minus_applic.upper;

    return {mu_minus_applic, mu_plus_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto MuBremsstrahlungModel::micro_xs(Applicability applic) const -> XsTable
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Interact with host data.
 */
void MuBremsstrahlungModel::step(CoreParams const& params,
                                 CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{MuBremsstrahlungExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void MuBremsstrahlungModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//!@}
//---------------------------------------------------------------------------//
}  // namespace celeritas
