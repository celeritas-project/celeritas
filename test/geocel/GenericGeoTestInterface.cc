//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestInterface.hh"

#include <stdexcept>
#include <gtest/gtest.h>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/Types.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"  // IWYU pragma: keep
#include "geocel/detail/LengthUnits.hh"
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
 * using a hardcoded value of 250 steps. (TestEm3 has more than 100.)
 */
auto GenericGeoTestInterface::track(Real3 const& pos, Real3 const& dir)
    -> TrackingResult
{
    int remaining_steps = 250;

    TrackingResult result;

    bool const check_surface_normal{this->supports_surface_normal()};
    if (!check_surface_normal)
    {
        static int warn_count{0};
        world_logger()(CELER_CODE_PROVENANCE,
                       warn_count++ == 0 ? LogLevel::warning : LogLevel::debug)
            << "Surface normal checking is disabled for "
            << this->gdml_basename() << " using " << this->geometry_type();
        result.disable_surface_normal();
    }

    CheckedGeoTrackView geo{
        this->make_geo_track_view_interface(),
        this->get_test_volumes(),
        this->geometry_interface(),
        this->unit_length(),
    };
    geo.check_normal(check_surface_normal);

    // Note: position is scaled according to test
    geo = this->make_initializer(pos, dir);
    // Checked track view prevents particles starting outside
    EXPECT_FALSE(geo.is_outside());

    auto const& vols = *this->get_test_volumes();
    bool const has_vol_inst = !vols.volume_instance_labels().empty();

    // Length scale and description
    auto const unit_length = this->unit_length();
    // Convert from Celeritas native unit system to unit test's internal system
    auto from_native_length
        = [scale = unit_length.value](auto&& v) { return v / scale; };
    // Bump tolerance is unitless
    SoftZero soft_zero{this->bump_tol() * unit_length.value};

    while (!geo.is_outside())
    {
        // Find next distance
        Propagation next;
        try
        {
            next = geo.find_next_step();
        }
        catch (std::exception const& e)
        {
            ADD_FAILURE() << "failed to find next step: " << e.what() << ": "
                          << geo;
            break;
        }
        if (!next.boundary)
        {
            ADD_FAILURE()
                << R"(failed to find the next boundary while inside the geometry )"
                << geo;
            result.volumes.push_back("[NO INTERCEPT]");
            break;
        }

        if (soft_zero(next.distance))
        {
            // Add the point to the bump list
            for (auto p : geo.pos())
            {
                result.bumps.push_back(from_native_length(p));
            }
            // Unscaled bump value is a nice separator and can hint where the
            // bump originates
            result.bumps.push_back(next.distance);
        }
        else
        {
            // Add distance and names
            result.distances.push_back(from_native_length(next.distance));
            result.volumes.emplace_back(test::volume_name(geo, vols));
            if (has_vol_inst)
            {
                result.volume_instances.emplace_back(
                    test::volume_instance_name(geo, vols));
            }

            real_type const half_distance = next.distance / 2;
            try
            {
                geo.move_internal(half_distance);
            }
            catch (std::exception const& e)
            {
                ADD_FAILURE()
                    << "failed movement of "
                    << from_native_length(half_distance) << ": " << e.what();
                break;
            }

            try
            {
                next = geo.find_next_step();
            }
            catch (std::exception const& e)
            {
                ADD_FAILURE() << "failed to find next step at "
                              << from_native_length(geo.pos()) << " along "
                              << geo.dir() << ": " << e.what();
                break;
            }
            EXPECT_SOFT_NEAR(next.distance, half_distance, soft_zero.abs());

            try
            {
                result.halfway_safeties.push_back(
                    from_native_length(geo.find_safety()));
            }
            catch (std::exception const& e)
            {
                ADD_FAILURE()
                    << "failed to find safety at "
                    << from_native_length(geo.pos()) << ": " << e.what();
                break;
            }

            if (!soft_zero(result.halfway_safeties.back()))
            {
                // Check reinitialization if not tangent to a surface
                GeoTrackInitializer const init{geo.pos(), geo.dir()};
                auto prev_id = geo.impl_volume_id();
                geo = init;
                if (geo.is_outside())
                {
                    ADD_FAILURE() << "reinitialization put the track outside "
                                     "the geometry "
                                  << geo;
                    break;
                }
                if (geo.impl_volume_id() != prev_id)
                {
                    ADD_FAILURE()
                        << "reinitialization changed the volume from "
                        << result.volumes.back() << " to "
                        << this->volume_name(geo) << " (alleged safety: "
                        << result.halfway_safeties.back() << " ["
                        << unit_length.label << "]) ";
                    result.volumes.back() += "/" + this->volume_name(geo);
                }
                next = geo.find_next_step();
                EXPECT_TRUE(next.boundary);
                EXPECT_SOFT_NEAR(next.distance, half_distance, soft_zero.abs())
                    << "reinitialized distance mismatch at index "
                    << result.volumes.size() - 1 << ": " << geo;
            }
        }

        // Move to the boundary and attempt to cross
        geo.move_to_boundary();
        try
        {
            geo.cross_boundary();
            if (check_surface_normal && !geo.is_outside())
            {
                // Add post-crossing (interior surface) dot product
                result.dot_normal.push_back([&] {
                    return std::fabs(dot_product(geo.dir(), geo.normal()));
                }());
            }
        }
        catch (std::exception const& e)
        {
            ADD_FAILURE() << "failed to cross boundary: " << e.what() << geo;
            break;
        }

        if (remaining_steps-- == 0)
        {
            ADD_FAILURE() << "maximum steps exceeded" << geo;
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
        this->get_test_volumes()->volume_instance_labels(),
        make_span(inst_ids));
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
 * Create a track view with runtime error checking.
 */
CheckedGeoTrackView GenericGeoTestInterface::make_checked_track_view()
{
    CheckedGeoTrackView result{this->make_geo_track_view_interface()};

    result.check_normal(this->supports_surface_normal());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Unit length for "track" testing and other results (defaults to cm).
 */
auto GenericGeoTestInterface::unit_length() const -> UnitLength
{
    return {};
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
    init.pos *= static_cast<real_type>(this->unit_length().value);
    return init;
}

//---------------------------------------------------------------------------//
std::string GenericGeoTestInterface::volume_name(GeoTrackView const& geo) const
{
    return ::celeritas::test::volume_name(geo, *this->get_test_volumes());
}

//---------------------------------------------------------------------------//
std::string
GenericGeoTestInterface::unique_volume_name(GeoTrackView const& geo) const
{
    return ::celeritas::test::unique_volume_name(geo,
                                                 *this->get_test_volumes());
}

//---------------------------------------------------------------------------//
auto GenericGeoTestInterface::get_test_volumes() const -> SPConstVolumes const&
{
    volumes_ = this->volumes();
    if (!volumes_)
    {
        // Built without using Geant4 model
        static PersistentSP<VolumeParams const> pv{
            "GenericGeoTestBase volumes"};
        pv.lazy_update(std::string{this->gdml_basename()},
                       [g = this->geometry_interface()]() {
                           return std::make_shared<VolumeParams const>(
                               g->make_model_input().volumes);
                       });
        volumes_ = pv.value();
    }
    CELER_ENSURE(volumes_);
    return volumes_;
}

}  // namespace test
}  // namespace celeritas
