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
#include "geocel/GeantGeoParams.hh"
#include "geocel/inp/Model.hh"

#include "CheckedGeoTrackView.hh"
#include "GenericGeoResults.hh"
#include "TestMacros.hh"
#include "UnitUtils.hh"

namespace celeritas
{
namespace test
{
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
 * ! Build the geometry (default to from_basename).
 */
template<class HP>
auto GenericGeoTestBase<HP>::build_geometry() -> SPConstGeo
{
    return this->build_geometry_from_basename();
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
    auto result = std::make_shared<HP>(test_file);
    if constexpr (std::is_same_v<HP, GeantGeoParams>)
    {
        // Save global geant geometry
        ::celeritas::geant_geo(*result);
    }
    return result;
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::geometry() -> SPConstGeo const&
{
    if (!geo_)
    {
        std::string key = this->geometry_basename() + "/"
                          + std::string{this->geometry_type()};
        // Construct via LazyGeoManager
        auto geo = this->get_geometry(key);
        EXPECT_TRUE(geo);
        geo_ = std::dynamic_pointer_cast<HP const>(geo);
        CELER_VALIDATE(geo_,
                       << "failed to cast geometry from "
                       << demangled_type(*geo) << " to "
                       << TypeDemangler<HP const>()());
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
    return this->geometry()->volumes().at(geo.volume_id()).name;
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
GenericGeoTestBase<HP>::all_volume_instance_names(GeoTrackView const& geo) const
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto level = geo.level();
    CELER_ASSERT(level && level >= LevelId{0});

    std::vector<VolumeInstanceId> ids(level.get() + 1);
    geo.volume_instance_id(make_span(ids));

    auto const& vol_inst = this->geometry()->volume_instances();
    std::ostringstream os;
    os << vol_inst.at(ids[0]);
    for (auto i : range(std::size_t{1}, ids.size()))
    {
        os << '/' << vol_inst.at(ids[i]);
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
    auto const& geo_params = *this->geometry();
    auto const& vol_inst = geo_params.volume_instances();
    real_type const inv_length = real_type{1} / this->unit_length();
    real_type const bump_tol = this->bump_tol() * this->unit_length();

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
        if (vol_inst)
        {
            result.volume_instances.push_back([&] {
                auto vi_id = geo.volume_instance_id();
                if (!vi_id)
                {
                    return std::string{"---"};
                }
                std::string s = vol_inst.at(vi_id).name;
                if (auto phys_inst = geo_params.id_to_geant(vi_id))
                {
                    if (phys_inst.replica)
                    {
                        s += '@';
                        s += std::to_string(phys_inst.replica.get());
                    }
                }
                return s;
            }());
        }
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
            if (vol_inst)
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

    return VolumeStackResult::from_span(*this->geometry(), make_span(inst_ids));
}

//---------------------------------------------------------------------------//
/*!
 * Get the model input from the geometry.
 */
template<class HP>
auto GenericGeoTestBase<HP>::model_inp() const -> ModelInpResult
{
    return ModelInpResult::from_model_input(
        this->geometry()->make_model_input());
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
auto GenericGeoTestBase<HP>::geometry_interface() const -> SPConstGeoInterface
{
    return this->geometry();
}

//---------------------------------------------------------------------------//
template<class HP>
auto GenericGeoTestBase<HP>::build_fresh_geometry(std::string_view)
    -> SPConstGeoI
{
    return this->build_geometry();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
