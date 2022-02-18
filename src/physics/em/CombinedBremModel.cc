//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CombinedBremModel.cc
//---------------------------------------------------------------------------//
#include "CombinedBremModel.hh"

#include <memory>
#include <utility>
#include "base/Assert.hh"
#include "base/Quantity.hh"
#include "physics/base/Applicability.hh"
#include "physics/em/RelativisticBremModel.hh"
#include "physics/em/SeltzerBergerModel.hh"
#include "physics/em/detail/CombinedBremData.hh"
#include "physics/em/detail/PhysicsConstants.hh"
#include "physics/em/detail/RelativisticBremData.hh"
#include "physics/em/detail/SeltzerBergerData.hh"
#include "physics/em/generated/CombinedBremInteract.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
CombinedBremModel::CombinedBremModel(ModelId               id,
                                     const ParticleParams& particles,
                                     const MaterialParams& materials,
                                     ReadData              sb_table,
                                     bool                  enable_lpm)
{
    CELER_EXPECT(id);
    CELER_EXPECT(sb_table);

    // Construct SeltzerBergerModel and RelativisticBremModel and save the
    // host data reference
    sb_model_ = std::make_shared<SeltzerBergerModel>(
        id, particles, materials, sb_table);

    rb_model_ = std::make_shared<RelativisticBremModel>(
        id, particles, materials, enable_lpm);

    detail::CombinedBremData<Ownership::value, MemSpace::host> host_ref;
    host_ref.sb_differential_xs = sb_model_->host_ref().differential_xs;
    host_ref.rb_data            = rb_model_->host_ref();

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<detail::CombinedBremData>{std::move(host_ref)};
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
    electron_brem.lower    = zero_quantity();
    electron_brem.upper    = detail::high_energy_limit();

    Applicability positron_brem = electron_brem;
    positron_brem.particle      = this->host_ref().rb_data.ids.positron;

    return {electron_brem, positron_brem};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void CombinedBremModel::interact(const DeviceInteractRef& data) const
{
    generated::combined_brem_interact(this->device_ref(), data);
}

void CombinedBremModel::interact(const HostInteractRef& data) const
{
    generated::combined_brem_interact(this->host_ref(), data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId CombinedBremModel::model_id() const
{
    return this->host_ref().rb_data.ids.model;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
