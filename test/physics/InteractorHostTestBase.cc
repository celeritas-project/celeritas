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
    ps_pointers_.vars = {&particle_state_, 1};
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
void InteractorHostTestBase::set_particle_params(
    const ParticleParams::VecAnnotatedDefs& defs)
{
    REQUIRE(!defs.empty());
    particle_params_ = std::make_shared<ParticleParams>(defs);
    pp_pointers_     = particle_params_->host_pointers();
}

//---------------------------------------------------------------------------//
/*!
 * Access particle parameters.
 */
const ParticleParams& InteractorHostTestBase::particle_params() const
{
    REQUIRE(particle_params_);
    return *particle_params_;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident particle data.
 */
void InteractorHostTestBase::set_inc_particle(PDGNumber pdg, real_type energy)
{
    REQUIRE(particle_params_);
    REQUIRE(pdg);
    REQUIRE(energy > 0);

    particle_state_.def_id = particle_params_->find(pdg);
    particle_state_.energy = energy;

    pt_view_ = std::make_shared<ParticleTrackView>(
        pp_pointers_, ps_pointers_, ThreadId{0});
}

//---------------------------------------------------------------------------//
/*!
 * Set an incident direction (and normalize it).
 */
void InteractorHostTestBase::set_inc_direction(const Real3& dir)
{
    REQUIRE(celeritas::norm(dir) > 0);

    inc_direction_ = dir;
    normalize_direction(&inc_direction_);
}

//---------------------------------------------------------------------------//
/*!
 * Resize secondaries.
 */
void InteractorHostTestBase::resize_secondaries(int count)
{
    REQUIRE(count > 0);
    secondaries_.resize(count);
    sa_view_ = std::make_shared<SecondaryAllocatorView>(
        secondaries_.host_pointers());
}

//---------------------------------------------------------------------------//
/*!
 * Check whether momentum is conserved in the interaction.
 */
void InteractorHostTestBase::check_conservation(const Interaction& interaction) const
{
    ParticleTrackState    local_state = particle_state_;
    ParticleStatePointers local_state_ptrs;
    local_state_ptrs.vars = {&local_state, 1};

    // Sum of exiting kinetic energy and momentum
    real_type exit_energy   = 0;
    Real3     exit_momentum = {0, 0, 0};

    // Subtract contribution from exiting particle state
    if (interaction && !action_killed(interaction.action))
    {
        local_state.def_id = particle_state_.def_id;
        local_state.energy = interaction.energy;
        ParticleTrackView exiting_track(
            pp_pointers_, local_state_ptrs, ThreadId{0});
        exit_energy += exiting_track.energy();
        axpy(exiting_track.momentum(), interaction.direction, &exit_momentum);
    }

    // Subtract contributions from exiting secondaries
    for (const Secondary& s : interaction.secondaries)
    {
        local_state.def_id = s.def_id;
        local_state.energy = s.energy;
        ParticleTrackView secondary_track(
            pp_pointers_, local_state_ptrs, ThreadId{0});
        exit_energy += secondary_track.energy();
        axpy(secondary_track.momentum(), s.direction, &exit_momentum);
    }

    // Compare against incident particle
    {
        ParticleTrackView parent_track(pp_pointers_, ps_pointers_, ThreadId{0});
        EXPECT_SOFT_EQ(parent_track.energy(), exit_energy);

        Real3 delta_momentum = exit_momentum;
        axpy(-parent_track.momentum(), inc_direction_, &delta_momentum);
        EXPECT_SOFT_EQ(0.0, dot_product(delta_momentum, delta_momentum))
            << "Incident: " << inc_direction_
            << " with p = " << parent_track.momentum()
            << "; exiting p = " << exit_momentum;
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
