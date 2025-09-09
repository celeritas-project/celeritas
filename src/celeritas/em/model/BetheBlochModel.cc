//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BetheBlochModel.cc
//---------------------------------------------------------------------------//
#include "BetheBlochModel.hh"

#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuHadIonizationData.hh"
#include "celeritas/em/distribution/BetheBlochEnergyDistribution.hh"
#include "celeritas/em/executor/MuHadIonizationExecutor.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

#include "detail/MuHadIonizationBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
BetheBlochModel::BetheBlochModel(ActionId id,
                                 ParticleParams const& particles,
                                 SetApplicability applicability)
    : StaticConcreteAction(
          id, "ioni-bethe-bloch", "interact by ionization (Bethe-Bloch)")
    , applicability_(applicability)
    , data_(detail::MuHadIonizationBuilder(particles,
                                           this->label())(applicability_))
{
    CELER_EXPECT(id);
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto BetheBlochModel::applicability() const -> SetApplicability
{
    return applicability_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto BetheBlochModel::micro_xs(Applicability) const -> XsTable
{
    // Aside from the production cut, the discrete interaction is material
    // independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void BetheBlochModel::step(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{MuHadIonizationExecutor<BetheBlochEnergyDistribution>{
            this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void BetheBlochModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
