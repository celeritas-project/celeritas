//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"

#include "CheckedGeoTrackView.hh"
#include "GenericGeoResults.hh"
#include "PersistentSP.hh"
#include "TestMacros.hh"
#include "UnitUtils.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Default constructor
template<class HP>
GenericGeoTestBase<HP>::GenericGeoTestBase() = default;

//---------------------------------------------------------------------------//
//! Anchored destructor
template<class HP>
GenericGeoTestBase<HP>::~GenericGeoTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Build geometry during setup.
 */
template<class HP>
void GenericGeoTestBase<HP>::SetUp()
{
    ASSERT_TRUE(this->geometry());
}

//---------------------------------------------------------------------------//
/*!
 * Return test suite name by default.
 */
template<class HP>
std::string_view GenericGeoTestBase<HP>::gdml_basename() const
{
    return ::testing::UnitTest::GetInstance()
        ->current_test_info()
        ->test_suite_name();
}

//---------------------------------------------------------------------------//
/*!
 * Build the geometry, defaulting to using the lazy Geant4 construction.
 */
template<class HP>
auto GenericGeoTestBase<HP>::build_geometry() const -> SPConstGeo
{
    auto geo_interface = this->lazy_geo();
    CELER_ASSERT(geo_interface);
    auto geo = std::dynamic_pointer_cast<HP const>(geo_interface);
    CELER_VALIDATE(geo,
                   << "failed to cast geometry from "
                   << demangled_type(*geo_interface) << " to "
                   << TypeDemangler<HP const>()());
    return geo;
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::geometry() -> SPConstGeo const&
{
    if (!geo_)
    {
        static PersistentSP<HP const> pg{"GenericGeoTestBase geometry"};

        auto basename = this->gdml_basename();
        pg.lazy_update(std::string{basename}, [this]() {
            // Build new geometry
            return this->build_geometry();
        });
        geo_ = pg.value();
        volumes_ = this->volumes();
        if (!volumes_)
        {
            // Built without using Geant4 model
            static PersistentSP<VolumeParams const> pv{
                "GenericGeoTestBase volumes"};
            pv.lazy_update(std::string{basename}, [&g = *geo_]() {
                return std::make_shared<VolumeParams const>(
                    g.make_model_input().volumes);
            });
            volumes_ = pv.value();
        }
    }
    CELER_ENSURE(geo_);
    CELER_ENSURE(volumes_);
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

    if (VolumeId id = geo.volume_id())
    {
        // Use volumes
        CELER_ASSERT(volumes_);
        return volumes_->volume_labels().at(id).name;
    }
    return "[INVALID]";
}

//---------------------------------------------------------------------------//
template<class HP>
std::string GenericGeoTestBase<HP>::surface_name(GeoTrackView const&) const
{
    // TODO: use Surfaces class
    return "---";
}

//---------------------------------------------------------------------------//
template<class HP>
std::string
GenericGeoTestBase<HP>::unique_volume_name(GeoTrackView const& geo) const
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto level = geo.level();
    CELER_ASSERT(level && level >= LevelId{0});

    std::vector<VolumeInstanceId> ids(level.get() + 1);
    geo.volume_instance_id(make_span(ids));

    CELER_ASSERT(volumes_);
    auto const& vol_inst = volumes_->volume_instance_labels();
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
    init.pos *= static_cast<real_type>(this->unit_length());
    geo = init;
    return geo;
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::track(Real3 const& pos, Real3 const& dir)
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

    GeoTrackView geo = CheckedGeoTrackView{this->make_geo_track_view(pos, dir)};
    CELER_ASSERT(volumes_);
    auto const& vol_inst = volumes_->volume_instance_labels();
    real_type const inv_length = real_type{1} / this->unit_length();
    real_type const bump_tol = this->bump_tol() * this->unit_length();

    // Cross boundary, checking and recording data
    auto cross_boundary = [&] {
        CELER_EXPECT(geo.is_on_boundary());

        std::optional<Real3> pre_norm;
        if (check_surface_normal && !geo.is_outside())
        {
            pre_norm = geo.normal();
        }

        geo.cross_boundary();
        EXPECT_TRUE(geo.is_on_boundary());

        if (check_surface_normal && !geo.is_outside())
        {
            auto post_norm = geo.normal();
            if (pre_norm)
            {
                CELER_ASSERT(!result.volumes.empty());

                auto post_vol = [&] {
                    auto vi_id = geo.volume_instance_id();
                    if (!vi_id)
                    {
                        return this->volume_name(geo);
                    }
                    return to_string(vol_inst.at(vi_id));
                }();

                // Not entering or exiting global; check direction similarity
                EXPECT_SOFT_NEAR(1.0,
                                 std::fabs(dot_product(*pre_norm, post_norm)),
                                 celeritas::sqrt_tol())
                    << "Normal is not consistent at boundary from "
                    << result.volume_instances.back() << " into " << post_vol
                    << ": previously " << repr(*pre_norm) << ", now "
                    << repr(post_norm);
                if (soft_zero(dot_product(geo.dir(), post_norm)))
                {
                    CELER_LOG(warning)
                        << "Crossed from " << result.volume_instances.back()
                        << " into " << post_vol
                        << " at a tangent; traveling along " << repr(geo.dir())
                        << ", normal is " << repr(post_norm);
                }
            }

            // Add post-crossing (interior surface) dot product
            result.dot_normal.push_back([&] {
                if (!geo.is_on_boundary())
                {
                    return TrackingResult::no_surface_normal;
                }
                return std::fabs(dot_product(geo.dir(), post_norm));
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
template<class HP>
auto GenericGeoTestBase<HP>::volume_stack(Real3 const& pos)
    -> VolumeStackResult
{
    auto geo = this->make_geo_track_view(pos, Real3{0, 0, 1});

    auto level = geo.level();
    if (!level)
    {
        return {};
    }
    std::vector<VolumeInstanceId> inst_ids(level.get() + 1);
    geo.volume_instance_id(make_span(inst_ids));

    CELER_ASSERT(volumes_);
    return VolumeStackResult::from_span(volumes_->volume_instance_labels(),
                                        make_span(inst_ids));
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for this geometry: Geant4, VecGeom, ORANGE.
 */
template<class HP>
std::string_view GenericGeoTestBase<HP>::geometry_type() const
{
    return TraitsT::name;
}

//---------------------------------------------------------------------------//
/*!
 * Access the geometry interface, building if needed.
 */
template<class HP>
auto GenericGeoTestBase<HP>::geometry_interface() const
    -> GeoParamsInterface const&
{
    auto result = this->geometry();
    CELER_ENSURE(result);
    return *result;
}

//---------------------------------------------------------------------------//
/*!
 * Build a new geometry via LazyGeantGeoManager.
 */
template<class HP>
auto GenericGeoTestBase<HP>::build_geo_from_geant(
    SPConstGeantGeo const& geant_geo) const -> SPConstGeoI
{
    CELER_EXPECT(geant_geo);
    return HP::from_geant(geant_geo);
}

//---------------------------------------------------------------------------//
/*!
 * Build a new geometry via LazyGeantGeoManager (fallback when no Geant4).
 */
template<class HP>
auto GenericGeoTestBase<HP>::build_geo_from_gdml(std::string const& filename) const
    -> SPConstGeoI
{
    CELER_EXPECT(!CELERITAS_USE_GEANT4);
    return HP::from_gdml(filename);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
