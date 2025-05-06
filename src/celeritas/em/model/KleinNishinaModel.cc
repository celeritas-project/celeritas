//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/KleinNishinaModel.cc
//---------------------------------------------------------------------------//
#include "KleinNishinaModel.hh"

#include "corecel/math/Quantity.hh"
#include "celeritas/em/executor/KleinNishinaExecutor.hh"
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
KleinNishinaModel::KleinNishinaModel(ActionId id,
                                     ParticleParams const& particles)
    : StaticConcreteAction(
          id,
          "scat-klein-nishina",
          R"(interact by Compton scattering (simple Klein-Nishina))")
{
    CELER_EXPECT(id);
    data_.ids.electron = particles.find(pdg::electron());
    data_.ids.gamma = particles.find(pdg::gamma());

    CELER_VALIDATE(data_.ids.electron && data_.ids.gamma,
                   << R"(missing electron and/or gamma particles (required for )"
                   << this->description() << ")");
    data_.inv_electron_mass = 1
                              / value_as<KleinNishinaData::Mass>(
                                  particles.get(data_.ids.electron).mass());
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto KleinNishinaModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = data_.ids.gamma;
    photon_applic.lower = zero_quantity();
    photon_applic.upper = max_quantity();

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto KleinNishinaModel::micro_xs(Applicability) const -> XsTable
{
    // Discrete interaction is material independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void KleinNishinaModel::step(CoreParams const& params,
                             CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{KleinNishinaExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void KleinNishinaModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
