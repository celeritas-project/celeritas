//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SurfaceFunctors.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/SurfaceFunctors.hh"

#include "orange/OrangeData.hh"
#include "orange/OrangeGeoTestBase.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/construct/SurfaceInputBuilder.hh"
#include "orange/surf/SurfaceAction.hh"
#include "orange/surf/Surfaces.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceFunctorsTest : public ::celeritas::test::OrangeGeoTestBase
{
  protected:
    void SetUp() override
    {
        UnitInput unit;
        unit.label = "dummy";

        {
            // Build surfaces
            SurfaceInputBuilder insert(&unit.surfaces);
            insert(PlaneX(1.25), "myplane");
            insert(Sphere({2.25, 1, 0}, 1.25), "sphere");
        }
        {
            // Create a volume
            VolumeInput v;
            v.logic = {0, 1, logic::lor, logic::ltrue, logic::lor};
            v.faces = {LocalSurfaceId{0}, LocalSurfaceId{1}};
            v.bbox = {{-1, -1, -1}, {1, 1, 1}};
            unit.volumes = {std::move(v)};
        }

        {
            unit.bbox = {{-1, -1, -1}, {1, 1, 1}};
        }

        // Construct a single dummy volume
        this->build_geometry(std::move(unit));

        auto const& host_ref = this->host_params();

        surfaces_ = std::make_unique<Surfaces>(
            host_ref, host_ref.simple_units[SimpleUnitId{0}].surfaces);
    }

    template<class T>
    T make_surface(size_type sid)
    {
        CELER_EXPECT(surfaces_);
        CELER_EXPECT(sid < surfaces_->num_surfaces());
        return surfaces_->make_surface<T>(LocalSurfaceId{sid});
    }

    std::unique_ptr<Surfaces> surfaces_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_sense)
{
    Real3 pos{0.9, 0, 0};
    CalcSense calc{pos};

    EXPECT_EQ(SignedSense::inside, calc(this->make_surface<PlaneX>(0)));
    EXPECT_EQ(SignedSense::outside, calc(this->make_surface<Sphere>(1)));

    pos = {1.0, 0, 0};
    EXPECT_EQ(SignedSense::inside, calc(this->make_surface<PlaneX>(0)));
    EXPECT_EQ(SignedSense::outside, calc(this->make_surface<Sphere>(1)));

    // Test as generic surfaces
    pos = {2, 0, 0};
    auto calc_generic = make_surface_action(*surfaces_, CalcSense{pos});
    EXPECT_EQ(SignedSense::outside, calc_generic(LocalSurfaceId{0}));
    EXPECT_EQ(SignedSense::inside, calc_generic(LocalSurfaceId{1}));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, num_intersections)
{
    auto num_intersections = make_static_surface_action<NumIntersections>();
    EXPECT_EQ(1, num_intersections(PlaneX::surface_type()));
    EXPECT_EQ(2, num_intersections(Sphere::surface_type()));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_normal)
{
    Real3 pos;
    auto calc_normal = make_surface_action(*surfaces_, CalcNormal{pos});

    pos = {1.25, 1, 1};
    EXPECT_EQ(Real3({1, 0, 0}), calc_normal(LocalSurfaceId{0}));
    pos = {2.25, 2.25, 0};
    EXPECT_EQ(Real3({0, 1, 0}), calc_normal(LocalSurfaceId{1}));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_safety_distance)
{
    Real3 pos;

    auto calc_distance
        = make_surface_action(*surfaces_, CalcSafetyDistance{pos});

    real_type eps = 1e-4;
    pos = {1.25 + eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(LocalSurfaceId{0}));
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(LocalSurfaceId{1}));

    pos = {1.25, 1, 0};
    EXPECT_SOFT_EQ(0, calc_distance(LocalSurfaceId{0}));
    EXPECT_SOFT_EQ(0.25, calc_distance(LocalSurfaceId{1}));

    pos = {1.25 - eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(LocalSurfaceId{0}));
    EXPECT_SOFT_EQ(0.25 - eps, calc_distance(LocalSurfaceId{1}));

    pos = {1.0 - eps, 1, 0};
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(LocalSurfaceId{0}));
    EXPECT_SOFT_EQ(eps, calc_distance(LocalSurfaceId{1}));

    pos = {3.5 + eps, 1, 0};
    EXPECT_SOFT_EQ(2.25 + eps, calc_distance(LocalSurfaceId{0}));
    EXPECT_SOFT_NEAR(0.0 + eps, calc_distance(LocalSurfaceId{1}), 1e-11);

    pos = {3.5, 1, 0};
    EXPECT_SOFT_EQ(2.25, calc_distance(LocalSurfaceId{0}));
    EXPECT_SOFT_EQ(0.0, calc_distance(LocalSurfaceId{1}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
