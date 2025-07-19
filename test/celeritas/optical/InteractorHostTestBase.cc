//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/InteractorHostTestBase.cc
//---------------------------------------------------------------------------//
#include "InteractorHostTestBase.hh"

#include "corecel/Assert.hh"
#include "corecel/math/ArrayUtils.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace celeritas::test;

//---------------------------------------------------------------------------//
/*!
 * Initialize the test base with simple values for the incident photon.
 */
InteractorHostBase::InteractorHostBase() : inc_direction_({0, 0, 1})
{
    ps_ = StateStore<ParticleStateData>(1);
    pt_view_ = std::make_shared<ParticleTrackView>(ps_.ref(), TrackSlotId{0});
    *pt_view_ = ParticleTrackView::Initializer{Energy{13e-6}, Real3{1, 0, 0}};

    ss_ = StateStore<SimStateData>(1);
    st_view_ = std::make_shared<SimTrackView>(ss_.ref(), TrackSlotId{0});
    *st_view_ = SimTrackView::Initializer{};
}

//---------------------------------------------------------------------------//
/*!
 * Get a random number generator with a clean counter.
 */
auto InteractorHostBase::rng() -> RandomEngine&
{
    rng_.reset_count();
    return rng_;
}

//---------------------------------------------------------------------------//
/*!
 * Set the direction of the incident photon.
 */
void InteractorHostBase::set_inc_direction(Real3 const& dir)
{
    inc_direction_ = dir;
}

//---------------------------------------------------------------------------//
/*!
 * Set the energy of the incident photon.
 */
void InteractorHostBase::set_inc_energy(Energy energy)
{
    CELER_EXPECT(pt_view_);
    ParticleTrackView::Initializer init;
    init.energy = energy;
    init.polarization = pt_view_->polarization();
    *pt_view_ = init;
}

//---------------------------------------------------------------------------//
/*!
 * Set the polarization of the incident photon.
 */
void InteractorHostBase::set_inc_polarization(Real3 const& pol)
{
    CELER_EXPECT(pt_view_);
    ParticleTrackView::Initializer init;
    init.energy = pt_view_->energy();
    init.polarization = pol;
    *pt_view_ = init;
}

//---------------------------------------------------------------------------//
/*!
 * Get the direction of the incident photon.
 */
auto InteractorHostBase::direction() const -> Real3 const&
{
    return inc_direction_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the track view of the incident photon.
 */
ParticleTrackView const& InteractorHostBase::particle_track() const
{
    CELER_EXPECT(pt_view_);
    return *pt_view_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the simulation track view.
 */
SimTrackView const& InteractorHostBase::sim_track() const
{
    CELER_EXPECT(st_view_);
    return *st_view_;
}

//---------------------------------------------------------------------------//
/*!
 * Check the direction and polarization are physical.
 *
 * The vectors should be unit vectors and perpendicular.
 */
void InteractorHostBase::check_direction_polarization(Real3 const& dir,
                                                      Real3 const& pol) const
{
    // Check vectors are unit vectors
    EXPECT_SOFT_EQ(1, norm(dir));
    EXPECT_SOFT_EQ(1, norm(pol));

    // Check direction and polarization are perpendicular
    EXPECT_SOFT_EQ(0, dot_product(dir, pol));
}

//---------------------------------------------------------------------------//
/*!
 * Check the direction and polarization of an interaction are physical.
 */
void InteractorHostBase::check_direction_polarization(
    Interaction const& interaction) const
{
    this->check_direction_polarization(interaction.direction,
                                       interaction.polarization);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
