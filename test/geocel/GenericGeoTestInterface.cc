//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestInterface.hh"

#include <algorithm>
#include <memory>
#include <random>
#include <variant>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/Types.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/CheckedGeoTrackView.hh"
#include "geocel/GeoInterface.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"  // IWYU pragma: keep
#include "geocel/inp/Model.hh"

#include "GenericGeoResults.hh"
#include "PersistentSP.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
void log_ggti_exception(
    LogProvenance where, char const* action, CheckedGeoError const& e)
{
    auto& log = self_logger();
    auto const& d = e.details();
    auto debug_msg = log({d.file, d.line}, LogLevel::debug);
    debug_msg << "Failed ";
    if (!d.condition.empty())
    {
        debug_msg << '\'' << d.condition << "' ";
    }
    debug_msg << "at " << d.file << ':' << d.line << " during '" << action
              << "'";

    // Log error message from originating call in `track`
    log(std::move(where), LogLevel::error) << "Failed: " << d.what;
}

//---------------------------------------------------------------------------//
struct StreamableActionException
{
    char const* action;
    CheckedGeoTrackView const& geo;
    std::exception const& e;

    [[maybe_unused]] friend std::ostream& operator<<(
        std::ostream& os, StreamableActionException const& sae)
    {
        os << "Caught exception during '" << sae.action
           << "': " << sae.e.what() << ": " << sae.geo;
        return os;
    }
};

//---------------------------------------------------------------------------//
//! Get the volume instances in the track's path
std::vector<VolumeInstanceId> get_path(GeoTrackInterface<real_type> const& geo)
{
    std::vector<VolumeInstanceId> result;
    if (!geo.is_outside())
    {
        geo.foreach_volume_path([&result](VolumeLevelId, VolumeInstanceId vi) {
            result.push_back(vi);
        });
    }
    return result;
}

//---------------------------------------------------------------------------//
//! Distance and path when leaving an ancestor
struct AncestorExit
{
    real_type distance{0};
    std::vector<VolumeInstanceId> path;
};

//---------------------------------------------------------------------------//
/*!
 * Track from volume to volume until leaving an ancestor.
 */
AncestorExit track_to_exit(
    CheckedGeoTrackView& geo, VolumeLevelId level, real_type max_distance)
{
    AncestorExit result;
    auto const start = get_path(geo);
    CELER_ASSERT(level < start.size());
    auto in_ancestor = [&](std::vector<VolumeInstanceId> const& path) {
        return path.size() > *level
               && std::equal(
                   start.begin(), start.begin() + *level + 1, path.begin());
    };

    int remaining{10000};
    do
    {
        result.distance += geo.find_next_step(max_distance).distance;
        geo.move_to_boundary();
        geo.cross_boundary();
        result.path = get_path(geo);
        CELER_VALIDATE(--remaining > 0, << "track is stuck: " << geo);
    } while (in_ancestor(result.path));
    return result;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Track until exiting the geometry.
 *
 * The position uses the length scale defined by the test. It is loop checked
 * using an input value.
 */
auto GenericGeoTestInterface::track(Real3 const& pos,
                                    Real3 const& dir,
                                    TrackingTol const& tol,
                                    int remaining_steps) -> TrackingResult
{
    TrackingResult result;
    CheckedGeoTrackView geo = this->make_checked_track_view();
    CELER_ASSERT(geo);
    if (!geo.check_normal())
    {
        static int warn_count{0};
        world_logger()(CELER_CODE_PROVENANCE,
                       warn_count++ == 0 ? LogLevel::warning : LogLevel::debug)
            << "Surface normal checking is disabled for "
            << this->gdml_basename() << " using " << this->geometry_type();
        result.disable_surface_normal();
    }

#define GGTI_EXPECT_NO_THROW(ACTION) \
    try \
    { \
        ACTION; \
    } \
    catch (CheckedGeoError const& e) \
    { \
        log_ggti_exception(CELER_CODE_PROVENANCE, #ACTION, e); \
        result.fail(); \
        return result; \
    } \
    catch (std::exception const& e) \
    { \
        ADD_FAILURE() << StreamableActionException{#ACTION, geo, e}; \
        result.fail(); \
        return result; \
    }

    // Note: position is scaled according to test
    GGTI_EXPECT_NO_THROW(geo = this->make_initializer(pos, dir));

    auto const& vols = *this->get_test_volumes();
    bool const has_vol_inst = !vols.volume_instance_labels().empty();

    // Length scale and description
    auto const unit_length = this->unit_length();
    // Convert from Celeritas native unit system to unit test's internal system
    auto from_native_length
        = [scale = unit_length.value](auto&& v) { return v / scale; };
    auto const& bbox = this->geometry_interface()->bbox();
    real_type const max_distance = distance(bbox.lower(), bbox.upper());

    while (!geo.is_outside())
    {
        // Find next distance
        Propagation next;
        GGTI_EXPECT_NO_THROW(next = geo.find_next_step(max_distance));

        if (SoftZero{tol.distance}(next.distance))
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
            result.volumes.emplace_back(celeritas::volume_name(geo, vols));
            if (has_vol_inst)
            {
                result.volume_instances.emplace_back(
                    celeritas::volume_instance_name(geo, vols));
            }

            // Move halfway to next boundary
            real_type const half_distance = next.distance / 2;
            GGTI_EXPECT_NO_THROW(geo.move_internal(half_distance));
            GGTI_EXPECT_NO_THROW(next = geo.find_next_step(max_distance));
            EXPECT_SOFT_NEAR(next.distance, half_distance, tol.distance) << geo;

            real_type safety{0};
            GGTI_EXPECT_NO_THROW(safety = geo.find_safety());
            result.halfway_safeties.push_back(from_native_length(safety));

            if (!SoftZero{tol.safety}(safety))
            {
                // Check reinitialization if not along a surface
                GeoTrackInitializer const init{geo.pos(), geo.dir()};
                auto prev_id = geo.impl_volume_id();
                GGTI_EXPECT_NO_THROW(geo = init);
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
                GGTI_EXPECT_NO_THROW(next = geo.find_next_step(max_distance));
                real_type length_scale = half_distance;
                for (auto x : geo.pos())
                {
                    length_scale = max(length_scale, std::fabs(x));
                }
                EXPECT_NEAR(
                    next.distance, half_distance, tol.distance * length_scale)
                    << "reinitialized distance mismatch (length scale="
                    << length_scale << ") at index "
                    << result.volumes.size() - 1 << ": " << geo;
            }
        }

        // Move to the boundary and attempt to cross
        GGTI_EXPECT_NO_THROW(geo.move_to_boundary());
        GGTI_EXPECT_NO_THROW(geo.cross_boundary());
        if (geo.check_normal() && !geo.is_outside())
        {
            Real3 normal{};
            GGTI_EXPECT_NO_THROW(normal = geo.normal());
            // Add post-crossing (interior surface) dot product
            result.dot_normal.push_back(
                std::fabs(dot_product(geo.dir(), normal)));
        }

        if (remaining_steps-- == 0)
        {
            ADD_FAILURE() << "maximum steps exceeded: " << geo;
            break;
        }
    }

    // Delete dot_normals that are all 1
    result.clear_boring_normals();

    return result;
}

// Track until exiting the geometry (default test tol)
auto GenericGeoTestInterface::track(
    Real3 const& pos_cm, Real3 const& dir, int max_steps) -> TrackingResult
{
    return this->track(pos_cm, dir, this->tracking_tol(), max_steps);
}

//---------------------------------------------------------------------------//
/*!
 * Check exiting and relocating inside every enclosing volume.
 *
 * Start random tracks in the geometry and compare against tracking from volume
 * to volume.
 */
void GenericGeoTestInterface::check_ancestor_exit(size_type num_tracks)
{
    auto const& bbox = this->geometry_interface()->bbox();
    real_type const max_distance = 2 * distance(bbox.lower(), bbox.upper());
    real_type const tol = 1e-6 * this->unit_length().value;

    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_type> sample_01{0, 1};
    auto sample_init = [&] {
        GeoTrackInitializer result;
        for (auto i : range(3))
        {
            // Stay off planes of symmetry and the world boundary
            result.pos[i]
                = bbox.lower()[i]
                  + (bbox.upper()[i] - bbox.lower()[i])
                        * (real_type(0.01) + real_type(0.98) * sample_01(rng));
            result.dir[i] = 2 * sample_01(rng) - 1;
        }
        result.dir = make_unit_vector(result.dir);
        return result;
    };

    // NOTE: track views created by this test share a track slot
    auto geo = this->make_checked_track_view();
    geo.check_normal(false);

    size_type num_daughter{0};
    for (auto i : range(num_tracks))
    {
        auto const init = sample_init();
        try
        {
            geo = init;
        }
        catch (CheckedGeoError const&)
        {
            // Inside the bounding box but outside the world
            continue;
        }
        auto const start_path = get_path(geo);
        for (auto lev : range(VolumeLevelId(start_path.size())))
        {
            SCOPED_TRACE(::testing::Message{}
                         << "track " << i << " in level " << lev.get()
                         << " at " << init.pos << " along " << init.dir);

            // Find the reference solution, including one step past the exit
            geo = init;
            auto const expected = track_to_exit(geo, lev, max_distance);
            AncestorExit expected_next;
            if (!geo.is_outside())
            {
                expected_next.distance
                    = geo.find_next_step(max_distance).distance;
                geo.move_to_boundary();
                geo.cross_boundary();
                expected_next.path = get_path(geo);
            }

            // Limit by a step that doesn't reach the boundary
            geo = init;
            auto next = geo.find_exit_step(lev, expected.distance / 2);
            EXPECT_FALSE(next.boundary);
            EXPECT_REAL_EQ(expected.distance / 2, next.distance);

            // Relocate inside the ancestor
            std::vector<AncestorExit> relocated;
            real_type moved{0};
            for (real_type frac : {0.05, 0.31, 0.5, 0.74, 0.93})
            {
                geo.move_relocate(lev, frac * expected.distance - moved);
                moved = frac * expected.distance;
                relocated.push_back({moved, get_path(geo)});

                next = geo.find_exit_step(lev, max_distance);
                EXPECT_TRUE(next.boundary);
                EXPECT_NEAR(expected.distance - moved, next.distance, tol);
            }

            // Move to the ancestor boundary
            geo.move_to_exit();
            EXPECT_TRUE(geo.is_on_boundary());
            auto path = get_path(geo);
            ASSERT_EQ(*lev + 1, path.size());
            EXPECT_TRUE(
                std::equal(path.begin(), path.end(), start_path.begin()));
            EXPECT_NEAR(expected.distance, distance(geo.pos(), init.pos), tol);

            // Leave the ancestor and take another step
            geo.cross_boundary();
            EXPECT_EQ(expected.path, get_path(geo));
            if (!geo.is_outside())
            {
                EXPECT_TRUE(geo.is_on_boundary());
                EXPECT_NEAR(expected_next.distance,
                            geo.find_next_step(max_distance).distance,
                            tol);
                geo.move_to_boundary();
                geo.cross_boundary();
                EXPECT_EQ(expected_next.path, get_path(geo));
            }

            // Check that the relocated states match an initialized state
            for (auto const& r : relocated)
            {
                geo = {init.pos + r.distance * init.dir, init.dir};
                EXPECT_EQ(get_path(geo), r.path) << geo;
                num_daughter += (r.path.size() > *lev + 1);
            }
        }
    }
    // Make sure we tested nontrivial cases
    EXPECT_GT(num_daughter, 0);
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance stack at a position.
 */
auto GenericGeoTestInterface::volume_stack(Real3 const& pos)
    -> VolumeStackResult
{
    VolumeStackResult result;

    CheckedGeoTrackView geo{this->make_geo_track_view_interface()};
    GGTI_EXPECT_NO_THROW(geo = this->make_initializer(pos, Real3{0, 0, 1}));

    auto vlev = geo.volume_level();
    if (!vlev)
    {
        return result;
    }
    std::vector<VolumeInstanceId> inst_ids(vlev.get() + 1);
    geo.volume_instance_id(make_span(inst_ids));

    result = VolumeStackResult::from_span(
        this->get_test_volumes()->volume_instance_labels(),
        make_span(inst_ids));

    return result;
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
    CheckedGeoTrackView result{
        this->make_geo_track_view_interface(),
        this->get_test_volumes(),
        this->geometry_interface(),
        this->unit_length(),
    };
    result.logger().level(getenv_loglevel("CELER_LOG_GEO", LogLevel::warning));
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
    result.normal = celeritas::sqrt_tol<real_type>;
    result.safety = result.distance;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an initializer with correct scaling/normalization.
 */
GeoTrackInitializer GenericGeoTestInterface::make_initializer(
    Real3 const& pos_unit_length, Real3 const& dir) const
{
    GeoTrackInitializer init{pos_unit_length, make_unit_vector(dir)};
    init.pos *= static_cast<real_type>(this->unit_length().value);
    return init;
}

//---------------------------------------------------------------------------//
std::string GenericGeoTestInterface::volume_name(GeoTrackView const& geo) const
{
    return ::celeritas::volume_name(geo, *this->get_test_volumes());
}

//---------------------------------------------------------------------------//
std::string GenericGeoTestInterface::unique_volume_name(
    GeoTrackView const& geo) const
{
    return ::celeritas::unique_volume_name(geo, *this->get_test_volumes());
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

        pv.lazy_update(
            std::string{this->gdml_basename()},
            [g = this->geometry_interface()]() {
                auto model_input = g->make_model_input();
                return std::visit(
                    Overload{[](inp::Volumes const& v) {
                                 return std::make_shared<VolumeParams const>(v);
                             },
                             [](std::shared_ptr<VolumeParams const> const& v) {
                                 return v;
                             }},
                    model_input.volumes);
            });
        volumes_ = pv.value();
    }
    CELER_ENSURE(volumes_);
    return volumes_;
}

}  // namespace test
}  // namespace celeritas
