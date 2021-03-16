//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InteractorHostTestBase.cc
//---------------------------------------------------------------------------//
#include "InteractorHostTestBase.hh"

#include "base/ArrayUtils.hh"
#include "base/StackAllocator.hh"
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
    ms_ = StateStore<celeritas::MaterialStateData>(*material_params_, 1);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident track's material
 */
void InteractorHostTestBase::set_material(const std::string& name)
{
    CELER_EXPECT(material_params_);

    mt_view_ = std::make_shared<MaterialTrackView>(
        material_params_->host_pointers(), ms_.ref(), ThreadId{0});

    // Initialize
    MaterialTrackView::Initializer_t init;
    init.material_id = material_params_->find(name);
    *mt_view_        = init;
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
    ps_ = StateStore<celeritas::ParticleStateData>(*particle_params_, 1);
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

    // Construct track view
    pt_view_ = std::make_shared<ParticleTrackView>(
        particle_params_->host_pointers(), ps_.ref(), ThreadId{0});

    // Initialize
    ParticleTrackView::Initializer_t init;
    init.particle_id = particle_params_->find(pdg);
    init.energy      = energy;
    *pt_view_        = init;
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
    secondaries_ = StateStore<SecondaryStackData>(count);
    sa_view_ = std::make_shared<StackAllocator<Secondary>>(secondaries_.ref());
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
    // Sum of exiting kinetic energy
    real_type exit_energy = interaction.energy_deposition.value();

    // Subtract contribution from exiting particle state
    if (interaction && !action_killed(interaction.action))
    {
        exit_energy += interaction.energy.value();
    }

    // Subtract contributions from exiting secondaries
    for (const Secondary& s : interaction.secondaries)
    {
        exit_energy += s.energy.value();
    }

    // Compare against incident particle
    EXPECT_SOFT_EQ(this->particle_track().energy().value(), exit_energy);
}

//---------------------------------------------------------------------------//
/*!
 * Check for momentum conservation in the interaction.
 */
void InteractorHostTestBase::check_momentum_conservation(
    const Interaction& interaction) const
{
    CollectionStateStore<celeritas::ParticleStateData, celeritas::MemSpace::host>
                      temp_store(*particle_params_, 1);
    ParticleTrackView temp_track(
        particle_params_->host_pointers(), temp_store.ref(), ThreadId{0});

    const auto& parent_track = this->particle_track();

    // Sum of exiting momentum
    Real3 exit_momentum = {0, 0, 0};

    // Subtract contribution from exiting particle state
    if (interaction && !action_killed(interaction.action))
    {
        ParticleTrackView::Initializer_t init;
        init.particle_id = parent_track.particle_id();
        init.energy      = interaction.energy;
        temp_track       = init;
        axpy(temp_track.momentum().value(),
             interaction.direction,
             &exit_momentum);
    }

    // Subtract contributions from exiting secondaries
    for (const Secondary& s : interaction.secondaries)
    {
        ParticleTrackView::Initializer_t init;
        init.particle_id = s.particle_id;
        init.energy      = s.energy;
        temp_track       = init;
        axpy(temp_track.momentum().value(), s.direction, &exit_momentum);
    }

    // Compare against incident particle
    {
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
