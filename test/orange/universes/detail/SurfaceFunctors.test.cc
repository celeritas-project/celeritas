//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/universes/detail/SurfaceFunctors.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/SurfaceFunctors.hh"

#include "orange/Data.hh"
#include "orange/construct/SurfaceInserter.hh"
#include "orange/surf/SurfaceAction.hh"
#include "orange/surf/Surfaces.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceFunctorsTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        SurfaceInserter insert(&surface_data_);
        insert(PlaneX(1.25));
        insert(Sphere({2.25, 1, 0}, 1.25));

        sd_ref_   = surface_data_;
        surfaces_ = std::make_unique<Surfaces>(sd_ref_);
    }

    template<class T>
    T make_surface(size_type sid)
    {
        CELER_EXPECT(surfaces_);
        CELER_EXPECT(sid < surfaces_->num_surfaces());
        return surfaces_->make_surface<T>(SurfaceId{sid});
    }

    SurfaceData<Ownership::value, MemSpace::host>           surface_data_;
    SurfaceData<Ownership::const_reference, MemSpace::host> sd_ref_;

    std::unique_ptr<Surfaces> surfaces_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_sense)
{
    using celeritas::detail::CalcSense;
    Real3     pos{0.9, 0, 0};
    CalcSense calc{pos};

    EXPECT_EQ(SignedSense::inside, calc(this->make_surface<PlaneX>(0)));
    EXPECT_EQ(SignedSense::outside, calc(this->make_surface<Sphere>(1)));

    pos = {1.0, 0, 0};
    EXPECT_EQ(SignedSense::inside, calc(this->make_surface<PlaneX>(0)));
    EXPECT_EQ(SignedSense::outside, calc(this->make_surface<Sphere>(1)));

    // Test as generic surfaces
    pos               = {2, 0, 0};
    auto calc_generic = make_surface_action(*surfaces_, CalcSense{pos});
    EXPECT_EQ(SignedSense::outside, calc_generic(SurfaceId{0}));
    EXPECT_EQ(SignedSense::inside, calc_generic(SurfaceId{1}));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, num_intersections)
{
    auto num_intersections
        = make_static_surface_action<celeritas::detail::NumIntersections>();
    EXPECT_EQ(1, num_intersections(PlaneX::surface_type()));
    EXPECT_EQ(2, num_intersections(Sphere::surface_type()));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_normal)
{
    Real3 pos;
    auto  calc_normal
        = make_surface_action(*surfaces_, celeritas::detail::CalcNormal{pos});

    pos = {1.25, 1, 1};
    EXPECT_EQ(Real3({1, 0, 0}), calc_normal(SurfaceId{0}));
    pos = {2.25, 2.25, 0};
    EXPECT_EQ(Real3({0, 1, 0}), calc_normal(SurfaceId{1}));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_safety_distance)
{
    Real3 pos;

    auto calc_distance = make_surface_action(
        *surfaces_, celeritas::detail::CalcSafetyDistance{pos});

    real_type eps = 1e-4;
    pos           = {1.25 + eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(SurfaceId{0}));
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(SurfaceId{1}));

    pos = {1.25, 1, 0};
    EXPECT_SOFT_EQ(0, calc_distance(SurfaceId{0}));
    EXPECT_SOFT_EQ(0.25, calc_distance(SurfaceId{1}));

    pos = {1.25 - eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(SurfaceId{0}));
    EXPECT_SOFT_EQ(0.25 - eps, calc_distance(SurfaceId{1}));

    pos = {1.0 - eps, 1, 0};
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(SurfaceId{0}));
    EXPECT_SOFT_EQ(eps, calc_distance(SurfaceId{1}));

    pos = {3.5 + eps, 1, 0};
    EXPECT_SOFT_EQ(2.25 + eps, calc_distance(SurfaceId{0}));
    EXPECT_SOFT_NEAR(0.0 + eps, calc_distance(SurfaceId{1}), 1e-11);

    pos = {3.5, 1, 0};
    EXPECT_SOFT_EQ(2.25, calc_distance(SurfaceId{0}));
    EXPECT_SOFT_EQ(0.0, calc_distance(SurfaceId{1}));
}
