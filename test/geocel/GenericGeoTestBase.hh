//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/data/CollectionStateStore.hh"
#include "geocel/GeoTraits.hh"
#include "geocel/WrappedGeoTrackView.hh"

#include "GenericGeoResults.hh"
#include "GenericGeoTestInterface.hh"
#include "LazyGeantGeoManager.hh"
#include "Test.hh"
#include "WrappedGeoTrackView.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Templated base class for loading geometry.
 *
 * \tparam G Geometry host params class, e.g. OrangeParams
 *
 * \note This class is instantiated in XTestBase.cc for geometry type X.
 */
template<class G>
class GenericGeoTestBase : virtual public Test, public GenericGeoTestInterface
{
    static_assert(std::is_base_of_v<GeoParamsInterface, G>);

    using TraitsT = GeoTraits<G>;

  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<G const>;
    using WrappedGeoTrack = WrappedGeoTrackView<typename TraitsT::TrackView>;
    //!@}

  public:
    // Default constructors and anchored destructor
    GenericGeoTestBase();
    virtual ~GenericGeoTestBase();
    CELER_DELETE_COPY_MOVE(GenericGeoTestBase);

    // Build geometry during setup
    void SetUp() override;

    //// Interface ////

    // Build the geometry for a new test (default to lazy geo)
    virtual SPConstGeo build_geometry() const;

    //// Geometry-specific functions ////

    // Build and/or access concrete (derived) geometry
    SPConstGeo const& geometry();
    SPConstGeo const& geometry() const;

    // Get a host track view
    WrappedGeoTrack make_geo_track_view(TrackSlotId tsid = TrackSlotId{0});
    //! Get and initialize a single-thread host track view
    WrappedGeoTrack make_geo_track_view(Real3 const& pos_cm, Real3 dir)
    {
        auto tv = this->make_geo_track_view();
        tv = this->make_initializer(pos_cm, dir);
        return tv;
    }

    //// GenericGeoTestInterface ////

    //! Create a track view
    UPGeoTrack make_geo_track_view_interface() final;
    // Get the label for this geometry: Geant4, VecGeom, ORANGE
    std::string_view geometry_type() const final;
    // Access the geometry interface
    SPConstGeoI geometry_interface() const final;

  private:
    template<Ownership W, MemSpace M>
    using StateData = typename TraitsT::template StateData<W, M>;
    using HostStateStore = CollectionStateStore<StateData, MemSpace::host>;

    SPConstGeo geo_;
    HostStateStore host_state_;

    //// LAZY GEO INTERFACE ////

    // Implementation builds from Geant4 on request
    [[nodiscard]] SPConstGeoI
    build_geo_from_geant(SPConstGeantGeo const&) const final;

    // Backup method when Geant4 is disabled
    [[nodiscard]] SPConstGeoI
    build_geo_from_gdml(std::string const& filename) const final;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
