//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "celeritas/em/generated/CombinedBremInteract.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"

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
{
    CELER_EXPECT(id);
    CELER_EXPECT(sb_table);

    // Construct SeltzerBergerModel and RelativisticBremModel and save the
    // host data reference
    sb_model_ = std::make_shared<SeltzerBergerModel>(
        id, particles, materials, data, sb_table);

    rb_model_ = std::make_shared<RelativisticBremModel>(
        id, particles, materials, data, enable_lpm);

    HostVal<CombinedBremData> host_ref;
    host_ref.ids.action = id;
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
    electron_brem.upper = high_energy_limit();

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
 * Apply the interaction kernel.
 */
void CombinedBremModel::execute(CoreDeviceRef const& data) const
{
    generated::combined_brem_interact(this->device_ref(), data);
}

void CombinedBremModel::execute(CoreHostRef const& data) const
{
    generated::combined_brem_interact(this->host_ref(), data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId CombinedBremModel::action_id() const
{
    return this->host_ref().rb_data.ids.action;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
