//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestBase.t.hh
//! \brief Templated definitions for GenericGeoTestBase
//---------------------------------------------------------------------------//
#pragma once

#include "GenericGeoTestBase.hh"

#include <memory>

#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/Types.hh"

#include "PersistentSP.hh"

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
auto GenericGeoTestBase<HP>::make_geo_track_view_interface() -> UPGeoTrack
{
    return std::make_unique<WrappedGeoTrack>(this->make_geo_track_view());
}

//---------------------------------------------------------------------------//
//! Get a host track view
template<class HP>
auto GenericGeoTestBase<HP>::make_geo_track_view(TrackSlotId tsid)
    -> WrappedGeoTrack
{
    if (!host_state_)
    {
        host_state_ = HostStateStore{this->geometry()->host_ref(),
                                     this->num_track_slots()};
    }
    CELER_EXPECT(tsid < host_state_.size());
    return WrappedGeoTrack{
        this->geometry()->host_ref(), host_state_.ref(), tsid};
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
