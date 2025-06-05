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

#include "GenericGeoTestInterface.hh"
#include "LazyGeoManager.hh"
#include "Test.hh"

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
 * \sa AllGeoTypedTestBase
 *
 * \note This class is instantiated in XTestBase.cc for geometry type X.
 */
template<class G>
class GenericGeoTestBase : virtual public Test,
                           public GenericGeoTestInterface,
                           private LazyGeoManager
{
    static_assert(std::is_base_of_v<GeoParamsInterface, G>);

    using TraitsT = GeoTraits<G>;

  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<G const>;
    using GeoTrackView = typename TraitsT::TrackView;
    //!@}

  public:
    // Build geometry during setup
    void SetUp() override;

    //// Interface ////

    // Build the geometry (default to from_basename)
    virtual SPConstGeo build_geometry();

    //! Maximum number of local track slots
    virtual size_type num_track_slots() const { return 1; }

    //// Geometry-specific functions ////

    //! Construct from celeritas test data and "basename" value
    SPConstGeo build_geometry_from_basename();

    // Access geometry
    SPConstGeo const& geometry();
    SPConstGeo const& geometry() const;

    //! Get the name of the current volume
    std::string volume_name(GeoTrackView const& geo) const;
    //! Get the name of the current surface if available
    virtual std::string surface_name(GeoTrackView const& geo) const;
    //! Get the stack of volume instances
    std::string all_volume_instance_names(GeoTrackView const& geo) const;

    //! Get a host track view
    GeoTrackView make_geo_track_view(TrackSlotId tsid = TrackSlotId{0});
    //! Get and initialize a single-thread host track view
    GeoTrackView make_geo_track_view(Real3 const& pos_cm, Real3 dir);

    //// GenericGeoTestInterface ////

    // Get the label for this geometry: Geant4, VecGeom, ORANGE
    std::string_view geometry_type() const final;
    // Access the geometry interface, building if needed
    SPConstGeoInterface geometry_interface() const final;
    // Find linear segments until outside
    TrackingResult track(Real3 const& pos_cm, Real3 const& dir) final;
    // Find linear segments until outside or max_step count is reached
    TrackingResult
    track(Real3 const& pos_cm, Real3 const& dir, int max_step) final;
    VolumeStackResult volume_stack(Real3 const& pos_cm) final;

  private:
    template<Ownership W, MemSpace M>
    using StateData = typename TraitsT::template StateData<W, M>;
    using HostStateStore = CollectionStateStore<StateData, MemSpace::host>;

    SPConstGeo geo_;
    HostStateStore host_state_;

    SPConstGeoI build_fresh_geometry(std::string_view) override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
