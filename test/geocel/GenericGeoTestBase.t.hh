//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestBase.t.hh
//! \brief Templated definitions for GenericGeoTestBase
//---------------------------------------------------------------------------//
#pragma once

#include "GenericGeoTestBase.hh"

#include <limits>

#include "corecel/math/ArrayUtils.hh"
#include "geocel/UnitUtils.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//
template<class HP>
auto GenericGeoTestBase<HP>::geometry_basename() const -> std::string
{
    // Get filename based on unit test name
    ::testing::TestInfo const* const test_info
        = ::testing::UnitTest::GetInstance()->current_test_info();
    CELER_ASSERT(test_info);
    return test_info->test_case_name();
}

//---------------------------------------------------------------------------//
//
template<class HP>
auto GenericGeoTestBase<HP>::build_geometry_from_basename() -> SPConstGeo
{
    // Construct filename:
    // ${SOURCE}/test/celeritas/data/${basename}${fileext}
    auto filename = this->geometry_basename() + std::string{TraitsT::ext};
    std::string test_file = test_data_path("geocel", filename);
    return std::make_shared<HP>(test_file);
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::geometry() -> SPConstGeo const&
{
    if (!geo_)
    {
        std::string key = this->geometry_basename() + "/"
                          + std::string{TraitsT::name};
        // Construct via LazyGeoManager
        geo_ = std::dynamic_pointer_cast<HP const>(this->get_geometry(key));
    }
    CELER_ENSURE(geo_);
    return geo_;
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::geometry() const -> SPConstGeo const&
{
    CELER_ENSURE(geo_);
    return geo_;
}

//---------------------------------------------------------------------------//
template<class HP>
std::string GenericGeoTestBase<HP>::volume_name(GeoTrackView const& geo) const
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }
    return this->geometry()->id_to_label(geo.volume_id()).name;
}

//---------------------------------------------------------------------------//
template<class HP>
std::string GenericGeoTestBase<HP>::surface_name(GeoTrackView const& geo) const
{
    if (!geo.is_on_boundary())
    {
        return "---";
    }

    auto* ptr = dynamic_cast<GeoParamsSurfaceInterface const*>(
        this->geometry().get());
    if (!ptr)
    {
        return "---";
    }

    // Only call this function if the geometry supports surfaces
    return ptr->id_to_label(geo.surface_id()).name;
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::make_geo_track_view() -> GeoTrackView
{
    if (!host_state_)
    {
        host_state_ = HostStateStore{this->geometry()->host_ref(), 1};
    }
    return GeoTrackView{
        this->geometry()->host_ref(), host_state_.ref(), TrackSlotId{0}};
}

//---------------------------------------------------------------------------//
// Get and initialize a single-thread host track view
template<class HP>
auto GenericGeoTestBase<HP>::make_geo_track_view(Real3 const& pos_cm, Real3 dir)
    -> GeoTrackView
{
    auto geo = this->make_geo_track_view();
    geo = GeoTrackInitializer{from_cm(pos_cm), make_unit_vector(dir)};
    return geo;
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::track(Real3 const& pos_cm, Real3 const& dir)
    -> TrackingResult
{
    return this->track(pos_cm, dir, std::numeric_limits<int>::max());
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::track(Real3 const& pos_cm,
                                   Real3 const& dir,
                                   int max_step) -> TrackingResult
{
    CELER_EXPECT(max_step > 0);
    TrackingResult result;

    GeoTrackView geo
        = CheckedGeoTrackView{this->make_geo_track_view(pos_cm, dir)};

    if (geo.is_outside())
    {
        // Initial step is outside but may approach inside
        result.volumes.push_back("[OUTSIDE]");
        auto next = geo.find_next_step();
        result.distances.push_back(to_cm(next.distance));
        if (next.boundary)
        {
            geo.move_to_boundary();
            geo.cross_boundary();
            EXPECT_TRUE(geo.is_on_boundary());
            --max_step;
        }
    }

    while (!geo.is_outside() && max_step > 0)
    {
        result.volumes.push_back(this->volume_name(geo));
        auto next = geo.find_next_step();
        result.distances.push_back(to_cm(next.distance));
        if (!next.boundary)
        {
            // Failure to find the next boundary while inside the geometry
            ADD_FAILURE();
            result.volumes.push_back("[NO INTERCEPT]");
            break;
        }
        if (next.distance > real_type(from_cm(1e-7)))
        {
            geo.move_internal(next.distance / 2);
            geo.find_next_step();
            result.halfway_safeties.push_back(to_cm(geo.find_safety()));

            if (result.halfway_safeties.back() > 0)
            {
                // Check reinitialization if not tangent to a surface
                GeoTrackInitializer const init{geo.pos(), geo.dir()};
                auto prev_id = geo.volume_id();
                geo = init;
                if (geo.is_outside())
                {
                    ADD_FAILURE() << "reinitialization put the track outside "
                                     "the geometry at"
                                  << init.pos;
                    break;
                }
                if (geo.volume_id() != prev_id)
                {
                    ADD_FAILURE()
                        << "reinitialization changed the volume at "
                        << init.pos << " along " << init.dir << " from "
                        << result.volumes.back() << " to "
                        << this->volume_name(geo) << " (alleged safety: "
                        << to_cm(result.halfway_safeties.back()) << ")";
                    result.volumes.back() += "/" + this->volume_name(geo);
                }
                auto new_next = geo.find_next_step();
                EXPECT_TRUE(new_next.boundary);
                EXPECT_SOFT_NEAR(new_next.distance, next.distance / 2, 1e-10)
                    << "reinitialized distance mismatch at index "
                    << result.volumes.size() - 1 << ": " << init.pos
                    << " along " << init.dir;
            }
        }
        geo.move_to_boundary();
        geo.cross_boundary();
        --max_step;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
