//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BetheBlochModel.cc
//---------------------------------------------------------------------------//
#include "BetheBlochModel.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/BetheBlochData.hh"
#include "celeritas/em/executor/BetheBlochExecutor.hh"
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
BetheBlochModel::BetheBlochModel(ActionId id,
                                 ParticleParams const& particles,
                                 SetApplicability applicability)
    : StaticConcreteAction(
        id, "ioni-bethe-bloch", "interact by ionization (Bethe-Bloch)")
    , applicability_(applicability)
{
    CELER_EXPECT(id);
    CELER_EXPECT(!applicability_.empty());

    HostVal<BetheBlochData> host_data;

    auto particle_ids = make_builder(&host_data.particles);
    particle_ids.reserve(applicability_.size());
    for (auto const& applic : applicability_)
    {
        CELER_VALIDATE(applic,
                       << "invalid applicability with particle ID "
                       << applic.particle.unchecked_get()
                       << " and energy limits ("
                       << value_as<units::MevEnergy>(applic.lower) << ", "
                       << value_as<units::MevEnergy>(applic.upper)
                       << ") [MeV] for Bethe-Bloch model");
        particle_ids.push_back(applic.particle);
    }

    host_data.electron = particles.find(pdg::electron());
    CELER_VALIDATE(host_data.electron,
                   << "missing electron (required for " << this->description()
                   << ")");

    host_data.electron_mass = particles.get(host_data.electron).mass();

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<BetheBlochData>{std::move(host_data)};

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
auto BetheBlochModel::micro_xs(Applicability) const -> MicroXsBuilders
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
        InteractionApplier{BetheBlochExecutor{this->host_ref()}});
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
