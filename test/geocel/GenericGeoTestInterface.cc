//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestInterface.hh"

#include <gtest/gtest.h>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"  // IWYU pragma: keep
#include "geocel/inp/Model.hh"

#include "CheckedGeoTrackView.hh"
#include "GenericGeoResults.hh"
#include "PersistentSP.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
/*!
 * Track until exiting the geometry.
 *
 * The position uses the length scale defined by the test. It is loop checked
 * using a hardcoded value of 1000 steps.
 */
auto GenericGeoTestInterface::track(Real3 const& pos, Real3 const& dir)
    -> TrackingResult
{
    int remaining_steps = 1000;

    TrackingResult result;

    bool const check_surface_normal{this->supports_surface_normal()};
    if (!check_surface_normal)
    {
        CELER_LOG(warning) << "Surface normal checking is disabled for "
                           << this->gdml_basename() << " using "
                           << this->geometry_type();
        result.disable_surface_normal();
    }

#if 0
    // FIXME: Geant4 solidstest fails at one point
    CheckedGeoTrackView geo{this->make_geo_track_view_interface()};
    geo.check_normal(check_surface_normal);
#else
    UPGeoTrack geo_sp = this->make_geo_track_view_interface();
    auto& geo = *geo_sp;
#endif

    // Note: position is scaled according to test
    geo = this->make_initializer(pos, dir);

    auto const& vol_inst = this->get_test_volumes().volume_instance_labels();
    real_type const inv_length = real_type{1} / this->unit_length();
    real_type const bump_tol = this->bump_tol() * this->unit_length();

    // Cross boundary, checking and recording data
    auto cross_boundary = [&] {
        CELER_EXPECT(geo.is_on_boundary());
        geo.cross_boundary();
        if (check_surface_normal && !geo.is_outside())
        {
            // Add post-crossing (interior surface) dot product
            result.dot_normal.push_back([&] {
                if (!geo.is_on_boundary())
                {
                    return TrackingResult::no_surface_normal;
                }
                return std::fabs(dot_product(geo.dir(), geo.normal()));
            }());
        }
    };

    if (geo.is_outside())
    {
        // Initial step is outside but may approach inside
        result.volumes.emplace_back(this->volume_name(geo));
        auto next = geo.find_next_step();
        result.distances.push_back(next.distance * inv_length);
        if (next.boundary)
        {
            geo.move_to_boundary();
            cross_boundary();
            --remaining_steps;
        }
    }

    while (!geo.is_outside())
    {
        // Add volume names
        result.volumes.emplace_back(this->volume_name(geo));
        if (!vol_inst.empty())
        {
            result.volume_instances.emplace_back([&] {
                VolumeInstanceId vi_id;
                try
                {
                    vi_id = geo.volume_instance_id();
                }
                catch (celeritas::DebugError const& e)
                {
                    std::ostringstream os;
                    os << "<exception at " << e.details().file << ':'
                       << e.details().line << ": " << e.details().condition
                       << '>';
                    return std::move(os).str();
                }
                if (!vi_id)
                {
                    return std::string{"---"};
                }
                return to_string(vol_inst.at(vi_id));
            }());
        }

        // Add next distance
        auto next = geo.find_next_step();
        result.distances.push_back(next.distance * inv_length);
        if (!next.boundary)
        {
            ADD_FAILURE() << "failed to find the next boundary while inside "
                             "the geometry";
            result.volumes.push_back("[NO INTERCEPT]");
            break;
        }
        if (next.distance < bump_tol)
        {
            // Don't add epsilon distances
            result.distances.pop_back();
            result.volumes.pop_back();
            if (!vol_inst.empty())
            {
                result.volume_instances.pop_back();
            }
            // Instead add the point to the bump list
            for (auto p : geo.pos())
            {
                result.bumps.push_back(p * inv_length);
            }
        }
        else
        {
            try
            {
                geo.move_internal(next.distance / 2);
            }
            catch (std::exception const& e)
            {
                ADD_FAILURE()
                    << "failed movement of " << next.distance * inv_length / 2
                    << " to " << geo.pos() * inv_length << " along "
                    << geo.dir() << ": " << e.what();
                break;
            }
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
                auto prev_id = geo.impl_volume_id();
                geo = init;
                if (geo.is_outside())
                {
                    ADD_FAILURE() << "reinitialization put the track outside "
                                     "the geometry at"
                                  << init.pos;
                    break;
                }
                if (geo.impl_volume_id() != prev_id)
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
            cross_boundary();
        }
        catch (std::exception const& e)
        {
            ADD_FAILURE() << "failed to cross boundary at "
                          << geo.pos() * inv_length << " along " << geo.dir()
                          << ": " << e.what();
            break;
        }

        if (remaining_steps-- == 0)
        {
            ADD_FAILURE() << "maximum steps exceeded";
            break;
        }
    }

    // Delete dot_normals that are all 1
    result.clear_boring_normals();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance stack at a position.
 */
auto GenericGeoTestInterface::volume_stack(Real3 const& pos)
    -> VolumeStackResult
{
    CheckedGeoTrackView geo{this->make_geo_track_view_interface()};
    geo = this->make_initializer(pos, Real3{0, 0, 1});

    auto vlev = geo.volume_level();
    if (!vlev)
    {
        return {};
    }
    std::vector<VolumeInstanceId> inst_ids(vlev.get() + 1);
    geo.volume_instance_id(make_span(inst_ids));

    return VolumeStackResult::from_span(
        this->get_test_volumes().volume_instance_labels(), make_span(inst_ids));
}

//---------------------------------------------------------------------------//
/*!
 * Return test suite name by default.
 */
std::string_view GenericGeoTestInterface::gdml_basename() const
{
    auto* ut = ::testing::UnitTest::GetInstance();
    CELER_ASSERT(ut);
    auto* test = ut->current_test_info();
    CELER_VALIDATE(
        test, << "cannot get default GDML filename when run outside test");
    return test->test_suite_name();
}

//---------------------------------------------------------------------------//
/*!
 * Unit length for "track" testing and other results (defaults to cm).
 */
Constant GenericGeoTestInterface::unit_length() const
{
    return lengthunits::centimeter;
}

//---------------------------------------------------------------------------//
size_type GenericGeoTestInterface::num_track_slots() const
{
    return 1;
}

//---------------------------------------------------------------------------//
/*!
 * Get the safety tolerance (defaults to SoftEq tol).
 */
GenericGeoTrackingTolerance GenericGeoTestInterface::tracking_tol() const
{
    GenericGeoTrackingTolerance result;
    result.distance = SoftEqual{}.rel();
    result.normal = celeritas::sqrt_tol();
    result.safety = result.distance;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Whether surface normals work for the current geometry/test.
 *
 * This defaults to true and should be disabled per geometry
 * implementation/geometry class.
 */
bool GenericGeoTestInterface::supports_surface_normal() const
{
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Get the threshold for a movement being a "bump".
 *
 * This unitless tolerance is multiplied by the test's unit length when used.
 */
real_type GenericGeoTestInterface::bump_tol() const
{
    return 1e-7;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an initializer with correct scaling/normalization.
 */
GeoTrackInitializer
GenericGeoTestInterface::make_initializer(Real3 const& pos_unit_length,
                                          Real3 const& dir) const
{
    GeoTrackInitializer init{pos_unit_length, make_unit_vector(dir)};
    init.pos *= static_cast<real_type>(this->unit_length());
    return init;
}

//---------------------------------------------------------------------------//
std::string GenericGeoTestInterface::volume_name(GeoTrackView const& geo) const
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    if (VolumeId id = geo.volume_id())
    {
        return this->get_test_volumes().volume_labels().at(id).name;
    }
    return "[INVALID]";
}

//---------------------------------------------------------------------------//
std::string
GenericGeoTestInterface::unique_volume_name(GeoTrackView const& geo) const
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto vlev = geo.volume_level();
    CELER_ASSERT(vlev && vlev >= VolumeLevelId{0});

    std::vector<VolumeInstanceId> ids(vlev.get() + 1);
    geo.volume_instance_id(make_span(ids));

    auto const& vol_inst = this->get_test_volumes().volume_instance_labels();
    std::ostringstream os;
    os << vol_inst.at(ids[0]);
    for (auto i : range(std::size_t{1}, ids.size()))
    {
        os << '/';
        if (ids[i])
        {
            os << vol_inst.at(ids[i]);
        }
        else
        {
            os << "[INVALID]";
        }
    }
    return std::move(os).str();
}

//---------------------------------------------------------------------------//
VolumeParams const& GenericGeoTestInterface::get_test_volumes() const
{
    volumes_ = this->volumes();
    if (!volumes_)
    {
        // Built without using Geant4 model
        static PersistentSP<VolumeParams const> pv{
            "GenericGeoTestBase volumes"};
        pv.lazy_update(std::string{this->gdml_basename()},
                       [&g = this->geometry_interface()]() {
                           return std::make_shared<VolumeParams const>(
                               g.make_model_input().volumes);
                       });
        volumes_ = pv.value();
    }
    CELER_ENSURE(volumes_);
    return *volumes_;
}

}  // namespace test
}  // namespace celeritas
