//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/SafetyCache.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/geo/SafetyCacheData.hh"
#include "celeritas/geo/SafetyCacheTrackView.hh"

#include "../AllGeoTypedTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

template<class HP>
class SafetyCacheTest : public AllGeoTypedTestBase<HP>
{
  protected:
    using SPConstGeo = typename GenericGeoTestBase<HP>::SPConstGeo;
    using GeoTrackView = typename GenericGeoTestBase<HP>::GeoTrackView;

    template<MemSpace M>
    using SafetyStateStore = CollectionStateStore<SafetyCacheStateData, M>;

    std::string geometry_basename() const final { return "simple-cms"; }

    //! Get a single-thread host track view for safety manipulation
    template<class GTV>
    decltype(auto) make_safety_track_view(GTV&& geo)
    {
        if (!host_state_)
        {
            host_state_ = HostStateStore{1};
        }
        return SafetyCacheTrackView{
            std::forward<GTV>(geo), host_state_.ref(), TrackSlotId{0}};
    }

    //! Get and initialize
    template<class GTV>
    decltype(auto) make_safety_track_view(GTV&& geo, bool use_safety)
    {
        auto result = this->make_safety_track_view(std::forward<GTV>(geo));
        result = SafetyCacheInitializer{use_safety};
        return result;
    }

  private:
    using HostStateStore = SafetyStateStore<MemSpace::host>;

    HostStateStore host_state_;
};

TYPED_TEST_SUITE(SafetyCacheTest, AllGeoTestingTypes, AllGeoTestingTypeNames);

//---------------------------------------------------------------------------//
// HOST TESTS
//----------------------------------------------------------------------------//

TYPED_TEST(SafetyCacheTest, nav_replacement)
{
    // Initialize at {1,0,0} along +y
    auto geo = this->make_geo_track_view({1, 0, 0}, {0, 1, 0});
    EXPECT_EQ("vacuum_tube", this->volume_name(geo));

    // Build and initialize safety
    auto sft = this->make_safety_track_view(geo, true);

    EXPECT_TRUE(sft.use_safety());

    // Check uninitialized safety
    EXPECT_DOUBLE_EQ(0, sft.safety());

    if (CELERITAS_DEBUG)
    {
        // Movement outside safety should be prohibited
        EXPECT_THROW(sft.move_internal(Real3{2, 3, 4}), DebugError);
        EXPECT_THROW(sft.move_to_boundary(), DebugError);
    }

    // Get safety up to a given distance (no nearby boundaries)
    // (returned safety may be more)
    sft.find_safety(20);
    EXPECT_LE(20, sft.safety());
    real_type const bonus_safety = sft.safety() - 20;

    // Move to another position within safety sphere
    Real3 pos{0.5, 19.0, 0.1};
    sft.move_internal(pos);
    EXPECT_VEC_SOFT_EQ(pos, sft.pos());
    EXPECT_VEC_SOFT_EQ(pos, geo.pos());

    EXPECT_SOFT_EQ(0.99315912625141323 + bonus_safety, sft.safety());

    // Recalculate safety with a long distance
    sft.find_safety(30);
    EXPECT_DOUBLE_EQ(10.993422191251788, sft.safety());

    // Find a boundary
    sft.set_dir({0, -1, 0});
    auto next = sft.find_next_step(5.0);
    EXPECT_FALSE(next.boundary);
    EXPECT_DOUBLE_EQ(5.0, next.distance);
    sft.move_internal({0, 15, 0});
    EXPECT_DOUBLE_EQ(6.9610531605205974, sft.safety());

    // Find and move to a boundary
    sft.set_dir({0, 1, 0});
    next = sft.find_next_step(25);
    EXPECT_TRUE(next.boundary);
    EXPECT_DOUBLE_EQ(15.0, next.distance);

    sft.move_to_boundary();
    EXPECT_DOUBLE_EQ(0, sft.safety());

    geo.cross_boundary();
    EXPECT_EQ("si_tracker", this->volume_name(geo));
}

TYPED_TEST(SafetyCacheTest, no_safety)
{
    auto geo = this->make_geo_track_view({29, 0, 0}, {0, 1, 0});
    auto sft = this->make_safety_track_view(geo, false);
    EXPECT_FALSE(sft.use_safety());

    if (CELERITAS_DEBUG)
    {
        // No safety-related operations should be permitted
        EXPECT_THROW(sft.safety(), DebugError);
        EXPECT_THROW(sft.find_safety(123), DebugError);
        EXPECT_THROW(sft.move_internal(Real3{2, 3, 4}), DebugError);
        EXPECT_THROW(sft.move_to_boundary(), DebugError);
    }
}

TYPED_TEST(SafetyCacheTest, persistence_and_const)
{
    {
        // Initialize at {29,0,0} along +y
        auto geo = this->make_geo_track_view({29, 0, 0}, {0, 1, 0});
        EXPECT_EQ("vacuum_tube", this->volume_name(geo));
        this->make_safety_track_view(geo, true);
    }
    {
        // Create safety view with rvalue geo track view
        auto sft = this->make_safety_track_view(this->make_geo_track_view());
        real_type safety = sft.find_safety(10);
        EXPECT_SOFT_EQ(safety, sft.safety());
        EXPECT_SOFT_EQ(1.0, sft.safety());
    }
    {
        // Create with const ref: *cannot* update safety but can access
        auto const geo = this->make_geo_track_view();
        auto sft = this->make_safety_track_view(geo);
        EXPECT_SOFT_EQ(1.0, sft.safety());
    }
}

TYPED_TEST(SafetyCacheTest, construction)
{
    using HostGeoStateStore =
        typename TestFixture::template GeoStateStore<MemSpace::host>;
    using HostSafetyStateStore =
        typename TestFixture::template SafetyStateStore<MemSpace::host>;
    using GeoTrackView = typename TestFixture::GeoTrackView;

    HostGeoStateStore primary_geo_data{this->geometry()->host_ref(), 1};
    HostSafetyStateStore primary_safety_data{1};

    GeoTrackView primary_geo{
        this->geometry()->host_ref(), primary_geo_data.ref(), TrackSlotId{0}};
    SafetyCacheTrackView primary_sft{
        primary_geo, primary_safety_data.ref(), TrackSlotId{0}};

    GeoTrackView secondary_geo = this->make_geo_track_view();
    SafetyCacheTrackView secondary_sft
        = this->make_safety_track_view(secondary_geo);

    double x = 10;
    for (auto primary_use_safety : {0, 1, 2})
    {
        SCOPED_TRACE(primary_use_safety == 0   ? "no parent safety"
                     : primary_use_safety == 1 ? "parent uncalculated safety"
                                               : "parent calculated safety");
        // Initialize primary and safety
        primary_geo = GeoTrackInitializer{{x, 0, 0}, {1, 0, 0}};
        primary_sft = SafetyCacheInitializer{primary_use_safety > 0};
        if (primary_use_safety == 2)
        {
            // Parent precalculates safety distance
            primary_sft.find_safety(25);
        }
        for (auto secondary_use_safety : {false, true})
        {
            SCOPED_TRACE(secondary_use_safety ? "secondary uses safety"
                                              : "no safety used for "
                                                "secondary");
            for (auto detailed : {false, true})
            {
                SCOPED_TRACE(detailed ? "initialization from other track slot"
                                      : "normal secondary initialization");
                if (detailed)
                {
                    using GeoDetailedInit =
                        typename GeoTrackView::DetailedInitializer;
                    using SafetyDetailedInit =
                        typename decltype(primary_sft)::DetailedInitializer;
                    secondary_geo = GeoDetailedInit{primary_geo, {0, 1, 0}};
                    secondary_sft = SafetyDetailedInit{primary_sft,
                                                       secondary_use_safety};
                    if (primary_use_safety == 2 && secondary_sft.use_safety())
                    {
                        EXPECT_SOFT_EQ(30 - x, secondary_sft.safety());
                    }
                }
                else
                {
                    secondary_geo = {{x, 0, 0}, {0, 1, 0}};
                    secondary_sft
                        = SafetyCacheInitializer{secondary_use_safety};
                }

                EXPECT_EQ(secondary_use_safety, secondary_sft.use_safety());
                if (secondary_sft.use_safety())
                {
                    secondary_sft.find_safety(15);
                    EXPECT_SOFT_EQ(30 - x, secondary_sft.safety());
                }

                x += 1.0;
                if (primary_use_safety == 2)
                {
                    primary_sft.move_internal({x, 0, 0});
                }
                else
                {
                    primary_geo.move_internal({x, 0, 0});
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
