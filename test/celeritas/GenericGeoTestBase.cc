//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GenericGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestBase.hh"

#include <limits>

#include "celeritas_config.h"
#if CELERITAS_USE_GEANT4
#    include <G4LogicalVolume.hh>
#endif

#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"
#include "celeritas/ext/GeantGeoUtils.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/io/ImportVolume.hh"

#include "UnitUtils.hh"
#include "geo/CheckedGeoTrackView.hh"

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
#include "TestMacros.hh"

using std::cout;
using namespace std::literals;
using GeantVolResult = celeritas::test::GenericGeoGeantImportVolumeResult;

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

//---------------------------------------------------------------------------//
void GeantVolResult::print_expected() const
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
GeantVolResult GeantVolResult::from_import(GeoParamsInterface const& geom,
                                           G4VPhysicalVolume const* world)
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

//---------------------------------------------------------------------------//
GeantVolResult
GeantVolResult::from_pointers([[maybe_unused]] GeoParamsInterface const& geom,
                              G4VPhysicalVolume const* world)
{
    CELER_VALIDATE(world, << "world volume is nullptr");
#if CELERITAS_USE_GEANT4
    using Result = GenericGeoGeantImportVolumeResult;
    Result result;
    for (G4LogicalVolume* lv : celeritas::geant_logical_volumes())
    {
        if (!lv)
        {
            result.volumes.push_back(Result::empty);
            continue;
        }
        auto id = geom.find_volume(lv);
        result.volumes.push_back(id ? static_cast<int>(id.unchecked_get())
                                    : Result::missing);
        if (!id)
        {
            result.missing_names.push_back(lv->GetName());
        }
    }
    return result;
#else
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

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
    std::string test_file = test_data_path("celeritas", filename);
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
                EXPECT_SOFT_NEAR(new_next.distance, next.distance / 2, 1e-10);
            }
        }
        geo.move_to_boundary();
        geo.cross_boundary();
        --max_step;
    }

    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT TEMPLATE INSTANTIATIONS
//---------------------------------------------------------------------------//
CELERTEST_INST_GEO(GenericGeoTestBase);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
