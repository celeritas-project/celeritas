//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 * Initialze the test base with simple values for the incident photon.
 */
InteractorHostTestBase::InteractorHostTestBase() : inc_direction_({0, 0, 1})
{
    pt_view_ = std::make_shared<TrackView>(Energy{13e-6}, Real3{1, 0, 0});
}

//---------------------------------------------------------------------------//
/*!
 * Get a random number generator with a clean counter.
 */
auto InteractorHostTestBase::rng() -> RandomEngine&
{
    rng_.reset_count();
    return rng_;
}

//---------------------------------------------------------------------------//
/*!
 * Set the direction of the incident photon.
 */
void InteractorHostTestBase::set_inc_direction(Real3 const& dir)
{
    inc_direction_ = dir;
}

//---------------------------------------------------------------------------//
/*!
 * Set the energy of the incident photon.
 */
void InteractorHostTestBase::set_inc_energy(Energy energy)
{
    CELER_EXPECT(pt_view_);
    pt_view_ = std::make_shared<TrackView>(energy, pt_view_->polarization());
}

//---------------------------------------------------------------------------//
/*!
 * Set the polarization of the incident photon.
 */
void InteractorHostTestBase::set_inc_polarization(Real3 const& pol)
{
    CELER_EXPECT(pt_view_);
    pt_view_ = std::make_shared<TrackView>(pt_view_->energy(), pol);
}

//---------------------------------------------------------------------------//
/*!
 * Get the direction of the incident photon.
 */
auto InteractorHostTestBase::direction() const -> Real3 const&
{
    return inc_direction_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the track view of the incident photon.
 */
TrackView const& InteractorHostTestBase::photon_track() const
{
    CELER_EXPECT(pt_view_);
    return *pt_view_;
}

//---------------------------------------------------------------------------//
/*!
 * Check the direction and polarization are physical.
 *
 * The vectors should be unit vectors and perpendicular.
 */
void InteractorHostTestBase::check_direction_polarization(Real3 const& dir,
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
void InteractorHostTestBase::check_direction_polarization(
    Interaction const& interaction) const
{
    this->check_direction_polarization(interaction.direction,
                                       interaction.polarization);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
