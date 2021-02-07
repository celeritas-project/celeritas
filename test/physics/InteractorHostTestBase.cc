//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InteractorHostTestBase.cc
//---------------------------------------------------------------------------//
#include "InteractorHostTestBase.hh"

#include "base/ArrayUtils.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "gtest/detail/Macros.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Initialize secondary allocation on construction.
 */
InteractorHostTestBase::InteractorHostTestBase()
{
    this->resize_secondaries(128);
    ps_pointers_.vars  = {&particle_state_, 1};
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
InteractorHostTestBase::~InteractorHostTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Set particle parameters.
 */
void InteractorHostTestBase::set_material_params(MaterialParams::Input inp)
{
    CELER_EXPECT(!inp.materials.empty());

    material_params_ = std::make_shared<MaterialParams>(std::move(inp));
    resize(&ms_data_, material_params_->host_pointers(), 1);
    ms_ref_ = ms_data_;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident track's material
 */
void InteractorHostTestBase::set_material(const std::string& name)
{
    CELER_EXPECT(material_params_);

    ThreadId tid{0};
    ms_data_.state[tid].material_id = material_params_->find(name);
    mt_view_                        = std::make_shared<MaterialTrackView>(
        material_params_->host_pointers(), ms_ref_, tid);
}

//---------------------------------------------------------------------------//
/*!
 * Access material parameters.
 */
const MaterialParams& InteractorHostTestBase::material_params() const
{
    CELER_EXPECT(material_params_);
    return *material_params_;
}

//---------------------------------------------------------------------------//
/*!
 * Set particle parameters.
 */
void InteractorHostTestBase::set_particle_params(ParticleParams::Input inp)
{
    CELER_EXPECT(!inp.empty());
    particle_params_ = std::make_shared<ParticleParams>(std::move(inp));
    pp_pointers_     = particle_params_->host_pointers();
}

//---------------------------------------------------------------------------//
/*!
 * Access particle parameters.
 */
const ParticleParams& InteractorHostTestBase::particle_params() const
{
    CELER_EXPECT(particle_params_);
    return *particle_params_;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident particle data
 */
void InteractorHostTestBase::set_inc_particle(PDGNumber pdg, MevEnergy energy)
{
    CELER_EXPECT(particle_params_);
    CELER_EXPECT(pdg);
    CELER_EXPECT(energy >= zero_quantity());

    particle_state_.particle_id = particle_params_->find(pdg);
    particle_state_.energy      = energy;

    pt_view_ = std::make_shared<ParticleTrackView>(
        pp_pointers_, ps_pointers_, ThreadId{0});
}

//---------------------------------------------------------------------------//
/*!
 * Set an incident direction (and normalize it).
 */
void InteractorHostTestBase::set_inc_direction(const Real3& dir)
{
    CELER_EXPECT(celeritas::norm(dir) > 0);

    inc_direction_ = dir;
    normalize_direction(&inc_direction_);
}

//---------------------------------------------------------------------------//
/*!
 * Resize secondaries.
 */
void InteractorHostTestBase::resize_secondaries(int count)
{
    CELER_EXPECT(count > 0);
    secondaries_.resize(count);
    sa_view_ = std::make_shared<SecondaryAllocatorView>(
        secondaries_.host_pointers());
}

//---------------------------------------------------------------------------//
/*!
 * Check for energy and momentum conservation in the interaction.
 */
void InteractorHostTestBase::check_conservation(const Interaction& interaction) const
{
    this->check_momentum_conservation(interaction);
    this->check_energy_conservation(interaction);
}

//---------------------------------------------------------------------------//
/*!
 * Check for energy conservation in the interaction.
 */
void InteractorHostTestBase::check_energy_conservation(
    const Interaction& interaction) const
{
    ParticleTrackState    local_state = particle_state_;
    ParticleStatePointers local_state_ptrs;
    local_state_ptrs.vars = {&local_state, 1};

    // Sum of exiting kinetic energy
    real_type exit_energy = interaction.energy_deposition.value();

    // Subtract contribution from exiting particle state
    if (interaction && !action_killed(interaction.action))
    {
        local_state.particle_id = particle_state_.particle_id;
        local_state.energy      = interaction.energy;
        ParticleTrackView exiting_track(
            pp_pointers_, local_state_ptrs, ThreadId{0});
        exit_energy += exiting_track.energy().value();
    }

    // Subtract contributions from exiting secondaries
    for (const Secondary& s : interaction.secondaries)
    {
        local_state.particle_id = s.particle_id;
        local_state.energy      = s.energy;
        ParticleTrackView secondary_track(
            pp_pointers_, local_state_ptrs, ThreadId{0});
        exit_energy += secondary_track.energy().value();
    }

    // Compare against incident particle
    {
        ParticleTrackView parent_track(pp_pointers_, ps_pointers_, ThreadId{0});
        EXPECT_SOFT_EQ(parent_track.energy().value(), exit_energy);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Check for momentum conservation in the interaction.
 */
void InteractorHostTestBase::check_momentum_conservation(
    const Interaction& interaction) const
{
    ParticleTrackState    local_state = particle_state_;
    ParticleStatePointers local_state_ptrs;
    local_state_ptrs.vars = {&local_state, 1};

    // Sum of exiting momentum
    Real3 exit_momentum = {0, 0, 0};

    // Subtract contribution from exiting particle state
    if (interaction && !action_killed(interaction.action))
    {
        local_state.particle_id = particle_state_.particle_id;
        local_state.energy      = interaction.energy;
        ParticleTrackView exiting_track(
            pp_pointers_, local_state_ptrs, ThreadId{0});
        axpy(exiting_track.momentum().value(),
             interaction.direction,
             &exit_momentum);
    }

    // Subtract contributions from exiting secondaries
    for (const Secondary& s : interaction.secondaries)
    {
        local_state.particle_id = s.particle_id;
        local_state.energy      = s.energy;
        ParticleTrackView secondary_track(
            pp_pointers_, local_state_ptrs, ThreadId{0});
        axpy(secondary_track.momentum().value(), s.direction, &exit_momentum);
    }

    // Compare against incident particle
    {
        ParticleTrackView parent_track(pp_pointers_, ps_pointers_, ThreadId{0});
        Real3             delta_momentum = exit_momentum;
        axpy(-parent_track.momentum().value(), inc_direction_, &delta_momentum);
        EXPECT_SOFT_NEAR(0.0,
                         dot_product(delta_momentum, delta_momentum),
                         parent_track.momentum().value() * 1e-12)
            << "Incident: " << inc_direction_
            << " with p = " << parent_track.momentum().value()
            << "* MeV/c; exiting p = " << exit_momentum;
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
