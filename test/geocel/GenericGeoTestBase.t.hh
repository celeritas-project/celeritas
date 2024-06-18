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

#include "corecel/math/ArrayOperators.hh"
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
    // ${SOURCE}/test/geocel/data/${basename}${fileext}
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
auto GenericGeoTestBase<HP>::make_geo_track_view(TrackSlotId tsid)
    -> GeoTrackView
{
    if (!host_state_)
    {
        host_state_ = HostStateStore{this->geometry()->host_ref(),
                                     this->num_track_slots()};
    }
    CELER_EXPECT(tsid < host_state_.size());
    return GeoTrackView{this->geometry()->host_ref(), host_state_.ref(), tsid};
}

//---------------------------------------------------------------------------//
// Get and initialize a single-thread host track view
template<class HP>
auto GenericGeoTestBase<HP>::make_geo_track_view(Real3 const& pos, Real3 dir)
    -> GeoTrackView
{
    auto geo = this->make_geo_track_view();
    GeoTrackInitializer init{pos, make_unit_vector(dir)};
    init.pos *= this->unit_length();
    geo = init;
    return geo;
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::track(Real3 const& pos, Real3 const& dir)
    -> TrackingResult
{
    return this->track(pos, dir, std::numeric_limits<int>::max());
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::track(Real3 const& pos,
                                   Real3 const& dir,
                                   int max_step) -> TrackingResult
{
    CELER_EXPECT(max_step > 0);
    TrackingResult result;

    GeoTrackView geo = CheckedGeoTrackView{this->make_geo_track_view(pos, dir)};
    real_type const inv_length = 1 / this->unit_length();

    if (geo.is_outside())
    {
        // Initial step is outside but may approach inside
        result.volumes.push_back("[OUTSIDE]");
        auto next = geo.find_next_step();
        result.distances.push_back(next.distance * inv_length);
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
        result.distances.push_back(next.distance * inv_length);
        if (!next.boundary)
        {
            ADD_FAILURE() << "failed to find the next boundary while inside "
                             "the geometry";
            result.volumes.push_back("[NO INTERCEPT]");
            break;
        }
        if (next.distance > real_type(from_cm(1e-7)))
        {
            geo.move_internal(next.distance / 2);
            try
            {
                geo.find_next_step();
            }
            catch (std::exception const& e)
            {
                ADD_FAILURE()
                    << "failed to find next step at " << geo.pos() * inv_length
                    << " along " << geo.dir() << ": " << e.what();
                break;
            }
            result.halfway_safeties.push_back(geo.find_safety() * inv_length);

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
                        << result.halfway_safeties.back() * inv_length << ")";
                    result.volumes.back() += "/" + this->volume_name(geo);
                }
                auto new_next = geo.find_next_step();
                EXPECT_TRUE(new_next.boundary);
                EXPECT_SOFT_NEAR(new_next.distance,
                                 next.distance / 2,
                                 100 * SoftEqual<>{}.rel())
                    << "reinitialized distance mismatch at index "
                    << result.volumes.size() - 1 << ": " << init.pos
                    << " along " << init.dir;
            }
        }
        geo.move_to_boundary();
        try
        {
            geo.cross_boundary();
        }
        catch (std::exception const& e)
        {
            ADD_FAILURE() << "failed to cross boundary at "
                          << geo.pos() * inv_length << " along " << geo.dir()
                          << ": " << e.what();
            break;
        }
        --max_step;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
