//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/CombinedBremModel.cc
//---------------------------------------------------------------------------//
#include "CombinedBremModel.hh"

#include <memory>
#include <type_traits>
#include <utility>

#include "corecel/math/Quantity.hh"
#include "celeritas/em/data/CombinedBremData.hh"
#include "celeritas/em/data/ElectronBremsData.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/em/data/SeltzerBergerData.hh"
#include "celeritas/em/executor/CombinedBremExecutor.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"

#include "RelativisticBremModel.hh"
#include "SeltzerBergerModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
CombinedBremModel::CombinedBremModel(ActionId id,
                                     ParticleParams const& particles,
                                     MaterialParams const& materials,
                                     SPConstImported data,
                                     ReadData sb_table,
                                     bool enable_lpm)
    : StaticConcreteAction(
          id,
          "brems-combined",
          R"(interact by bremsstrahlung (combined SB/relativistic, e+/-))")
{
    CELER_EXPECT(id);
    CELER_EXPECT(sb_table);

    // Construct SeltzerBergerModel and RelativisticBremModel and save the
    // host data reference
    sb_model_ = std::make_shared<SeltzerBergerModel>(
        id, particles, materials, data, sb_table);

    rb_model_ = std::make_shared<RelativisticBremModel>(
        id, particles, materials, data, enable_lpm);

    CELER_VALIDATE(sb_model_->host_ref().high_energy_limit
                       == rb_model_->host_ref().low_energy_limit,
                   << "Seltzer-Berger high energy limit ("
                   << sb_model_->host_ref().high_energy_limit.value()
                   << " MeV) is inconsistent with relativistic "
                      "bremsstrahlung low energy limit ("
                   << rb_model_->host_ref().low_energy_limit.value()
                   << " MeV)");

    HostVal<CombinedBremData> host_ref;
    host_ref.sb_differential_xs = sb_model_->host_ref().differential_xs;
    host_ref.rb_data = rb_model_->host_ref();

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<CombinedBremData>{std::move(host_ref)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto CombinedBremModel::applicability() const -> SetApplicability
{
    Applicability electron_brem;
    electron_brem.particle = this->host_ref().rb_data.ids.electron;
    electron_brem.lower = zero_quantity();
    electron_brem.upper = detail::high_energy_limit();

    Applicability positron_brem = electron_brem;
    positron_brem.particle = this->host_ref().rb_data.ids.positron;

    return {electron_brem, positron_brem};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto CombinedBremModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Multiple elements per material not supported for combined brems model
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Interact with host data.
 */
void CombinedBremModel::step(CoreParams const& params,
                             CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{CombinedBremExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void CombinedBremModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
