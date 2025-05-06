//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MollerBhabhaModel.cc
//---------------------------------------------------------------------------//
#include "MollerBhabhaModel.hh"

#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MollerBhabhaData.hh"
#include "celeritas/em/executor/MollerBhabhaExecutor.hh"
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
MollerBhabhaModel::MollerBhabhaModel(ActionId id,
                                     ParticleParams const& particles)
    : StaticConcreteAction(
          id, "ioni-moller-bhabha", "interact by Moller+Bhabha ionization")
{
    CELER_EXPECT(id);
    data_.ids.electron = particles.find(pdg::electron());
    data_.ids.positron = particles.find(pdg::positron());

    CELER_VALIDATE(
        data_.ids.electron && data_.ids.positron,
        << R"(missing electron and/or positron particles (required for )"
        << this->description() << ")");

    data_.electron_mass = particles.get(data_.ids.electron).mass();

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MollerBhabhaModel::applicability() const -> SetApplicability
{
    /*!
     * \todo Set lower energy bound based on (material-dependent)
     * IonizationProcess lambda table energy grid to avoid invoking the
     * interactor for tracks with energy below the interaction threshold.
     */

    Applicability electron_applic, positron_applic;

    electron_applic.particle = data_.ids.electron;
    electron_applic.lower = zero_quantity();
    electron_applic.upper = units::MevEnergy{data_.max_valid_energy()};

    positron_applic.particle = data_.ids.positron;
    positron_applic.lower = zero_quantity();
    positron_applic.upper = electron_applic.upper;

    return {electron_applic, positron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto MollerBhabhaModel::micro_xs(Applicability) const -> XsTable
{
    // Aside from the production cut, the discrete interaction is material
    // independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void MollerBhabhaModel::step(CoreParams const& params,
                             CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{MollerBhabhaExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void MollerBhabhaModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
