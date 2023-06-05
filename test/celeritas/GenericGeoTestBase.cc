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
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/io/ImportVolume.hh"

#if CELERITAS_USE_VECGEOM
#    include "celeritas/ext/VecgeomData.hh"
#    include "celeritas/ext/VecgeomParams.hh"
#    include "celeritas/ext/VecgeomTrackView.hh"
#endif
#if CELERITAS_USE_GEANT4
#    include "celeritas/ext/GeantGeoData.hh"
#    include "celeritas/ext/GeantGeoParams.hh"
#    include "celeritas/ext/GeantGeoTrackView.hh"
#endif

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void GenericGeoTrackingResult::print_expected()
{
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

void GenericGeoGeantImportVolumeResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static int const expected_volumes[] = "
         << repr(this->volumes)
         << ";\n"
            "EXPECT_VEC_EQ(expected_volumes, result.volumes);\n"
            "EXPECT_EQ(0, result.missing_names.size()) << "
            "repr(result.missing_names);\n";
    if (!this->missing_names.empty())
    {
        cout << "/* Currently missing: " << repr(this->missing_names)
             << " */\n";
    }
    cout << "/*** END CODE ***/\n";
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
// Get and initialize a single-thread host track view
template<class HP, template<Ownership, MemSpace> class S, class TV>
auto GenericGeoTestBase<HP, S, TV>::calc_bump_pos(GeoTrackView const& geo,
                                                  real_type delta) const
    -> Real3
{
    CELER_EXPECT(delta > 0);
    auto pos = geo.pos();
    axpy(delta, geo.dir(), &pos);
    return pos;
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
        if (next.distance > real_type(1e-7))
        {
            geo.move_internal(next.distance / 2);
            geo.find_next_step();
            result.halfway_safeties.push_back(geo.find_safety());
        }
        geo.move_to_boundary();
        geo.cross_boundary();
    }

    return result;
}

//---------------------------------------------------------------------------//
namespace
{
// Non-templated helper function since this uses the virtual geo interface
GenericGeoGeantImportVolumeResult
get_geant_volumes_impl(G4VPhysicalVolume const* world,
                       GeoParamsInterface const& geom)
{
    CELER_VALIDATE(world, << "world volume is nullptr");
    // Load geometry before checking world volume
    GeantImporter import(world);
    auto imported = import([] {
        GeantImportDataSelection select;
        select.particles = GeantImportDataSelection::none;
        select.processes = GeantImportDataSelection::none;
        select.materials = false;
        select.reader_data = false;
        select.unique_volumes = true;  // emulates accel/SharedParams
        return select;
    }());

    using Result = GenericGeoGeantImportVolumeResult;
    Result result;
    for (auto i : range(imported.volumes.size()))
    {
        ImportVolume const& v = imported.volumes[i];
        if (v.name.empty())
        {
            // Add a placeholder only if it's not a leading "empty" (probably
            // indicative of unused 'instance IDs' from a previously loaded
            // geometry)
            if (!result.volumes.empty())
            {
                result.volumes.push_back(Result::empty);
            }
            continue;
        }
        auto id = geom.find_volume(Label::from_geant(v.name));
        result.volumes.push_back(id ? static_cast<int>(id.unchecked_get())
                                    : Result::missing);
        if (!id)
        {
            result.missing_names.push_back(
                to_string(Label::from_geant(v.name)));
        }
    }
    return result;
}
}  // namespace

//---------------------------------------------------------------------------//
template<class HP, template<Ownership, MemSpace> class S, class TV>
auto GenericGeoTestBase<HP, S, TV>::get_geant_volumes(
    G4VPhysicalVolume const* world) -> GeantVolResult
{
    return get_geant_volumes_impl(world, *this->geometry());
}

//---------------------------------------------------------------------------//
// EXPLICIT TEMPLATE INSTANTIATIONS
//---------------------------------------------------------------------------//
template class GenericGeoTestBase<OrangeParams, OrangeStateData, OrangeTrackView>;
#if CELERITAS_USE_VECGEOM
template class GenericGeoTestBase<VecgeomParams, VecgeomStateData, VecgeomTrackView>;
#endif

#if CELERITAS_USE_GEANT4
template class GenericGeoTestBase<GeantGeoParams,
                                  GeantGeoStateData,
                                  GeantGeoTrackView>;
#endif
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
