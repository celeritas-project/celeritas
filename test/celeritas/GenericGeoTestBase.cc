//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GenericGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestBase.hh"

#include "celeritas_config.h"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"

#if CELERITAS_USE_VECGEOM
#    include "celeritas/ext/VecgeomData.hh"
#    include "celeritas/ext/VecgeomParams.hh"
#    include "celeritas/ext/VecgeomTrackView.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void GenericGeoTrackingResult::print_expected()
{
    using std::cout;
    cout
        << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
           "static char const* const expected_volumes[] = "
        << repr(this->volumes)
        << ";\n"
           "EXPECT_VEC_EQ(expected_volumes, result.volumes);\n"
           "static real_type const expected_distances[] = "
        << repr(this->distances)
        << ";\n"
           "EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);\n"
           "static real_type const expected_hw_safety[] = "
        << repr(this->halfway_safeties)
        << ";\n"
           "EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);\n"
           "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
template<class HP, template<Ownership, MemSpace> class S, class TV>
auto GenericGeoTestBase<HP, S, TV>::geometry() -> SPConstGeo const&
{
    if (!geo_)
    {
        // Get filename based on unit test name
        ::testing::TestInfo const* const test_info
            = ::testing::UnitTest::GetInstance()->current_test_info();
        CELER_ASSERT(test_info);

        // Construct via LazyGeoManager
        geo_ = std::dynamic_pointer_cast<HP const>(
            this->get_geometry(test_info->test_case_name()));
    }
    CELER_ENSURE(geo_);
    return geo_;
}

//---------------------------------------------------------------------------//
template<class HP, template<Ownership, MemSpace> class S, class TV>
auto GenericGeoTestBase<HP, S, TV>::geometry() const -> SPConstGeo const&
{
    CELER_ENSURE(geo_);
    return geo_;
}

//---------------------------------------------------------------------------//
template<class HP, template<Ownership, MemSpace> class S, class TV>
std::string
GenericGeoTestBase<HP, S, TV>::volume_name(GeoTrackView const& geo) const
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }
    return this->geometry()->id_to_label(geo.volume_id()).name;
}

//---------------------------------------------------------------------------//
template<class HP, template<Ownership, MemSpace> class S, class TV>
std::string
GenericGeoTestBase<HP, S, TV>::surface_name(GeoTrackView const& geo) const
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
template<class HP, template<Ownership, MemSpace> class S, class TV>
auto GenericGeoTestBase<HP, S, TV>::make_geo_track_view() -> GeoTrackView
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
template<class HP, template<Ownership, MemSpace> class S, class TV>
auto GenericGeoTestBase<HP, S, TV>::make_geo_track_view(Real3 const& pos,
                                                        Real3 dir)
    -> GeoTrackView
{
    normalize_direction(&dir);

    auto geo = this->make_geo_track_view();
    geo = GeoTrackInitializer{pos, dir};
    return geo;
}

//---------------------------------------------------------------------------//
template<class HP, template<Ownership, MemSpace> class S, class TV>
auto GenericGeoTestBase<HP, S, TV>::track(Real3 const& pos, Real3 const& dir)
    -> TrackingResult
{
    TrackingResult result;

    GeoTrackView geo = this->make_geo_track_view(pos, dir);

    if (geo.is_outside())
    {
        // Initial step is outside but may approach insidfe
        result.volumes.push_back("[OUTSIDE]");
        auto next = geo.find_next_step();
        result.distances.push_back(next.distance);
        if (next.boundary)
        {
            geo.move_to_boundary();
            geo.cross_boundary();
            EXPECT_TRUE(geo.is_on_boundary());
        }
    }

    while (!geo.is_outside())
    {
        result.volumes.push_back(this->volume_name(geo));
        auto next = geo.find_next_step();
        result.distances.push_back(next.distance);
        if (!next.boundary)
        {
            // Failure to find the next boundary while inside the geometry
            ADD_FAILURE();
            result.volumes.push_back("[NO INTERCEPT]");
            break;
        }
        geo.move_internal(next.distance / 2);
        result.halfway_safeties.push_back(geo.find_safety());
        geo.move_to_boundary();
        geo.cross_boundary();
    }

    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT TEMPLATE INSTANTIATIONS
//---------------------------------------------------------------------------//
template class GenericGeoTestBase<OrangeParams, OrangeStateData, OrangeTrackView>;
#if CELERITAS_USE_VECGEOM
template class GenericGeoTestBase<VecgeomParams, VecgeomStateData, VecgeomTrackView>;
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
