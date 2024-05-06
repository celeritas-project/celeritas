//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/data/CollectionStateStore.hh"
#include "geocel/GeoTraits.hh"
#include "geocel/detail/LengthUnits.hh"

#include "LazyGeoManager.hh"
#include "Test.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct GenericGeoTrackingResult
{
    std::vector<std::string> volumes;
    std::vector<real_type> distances;  //!< [cm]
    std::vector<real_type> halfway_safeties;  //!< [cm]

    void print_expected();
};

//---------------------------------------------------------------------------//
struct GenericGeoGeantImportVolumeResult
{
    static constexpr int empty = -1;
    static constexpr int missing = -2;

    static GenericGeoGeantImportVolumeResult
    from_import(GeoParamsInterface const& geom, G4VPhysicalVolume const* world);

    static GenericGeoGeantImportVolumeResult
    from_pointers(GeoParamsInterface const& geom,
                  G4VPhysicalVolume const* world);

    std::vector<int> volumes;  //!< Volume ID for each Geant4 instance ID
    std::vector<std::string> missing_names;  //!< G4LV names without a match

    void print_expected() const;
};

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
class GenericGeoTestBase : virtual public Test, private LazyGeoManager
{
    static_assert(std::is_base_of_v<GeoParamsInterface, G>);

    using TraitsT = GeoTraits<G>;

  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<G const>;
    using GeoTrackView = typename TraitsT::TrackView;
    using TrackingResult = GenericGeoTrackingResult;
    using GeantVolResult = GenericGeoGeantImportVolumeResult;
    //!@}

  public:
    //! Get the basename or unique geometry key (defaults to suite name)
    virtual std::string geometry_basename() const;

    //! Build the geometry
    virtual SPConstGeo build_geometry() = 0;

    //! Maximum number of local track slots
    virtual size_type num_track_slots() const { return 1; }

    //! Unit length for "track" testing and other results
    virtual real_type unit_length() const { return lengthunits::centimeter; }

    //! Construct from celeritas test data and "basename" value
    SPConstGeo build_geometry_from_basename();

    // Access geometry
    SPConstGeo const& geometry();
    SPConstGeo const& geometry() const;

    //! Get the name of the current volume
    std::string volume_name(GeoTrackView const& geo) const;
    //! Get the name of the current surface if available
    std::string surface_name(GeoTrackView const& geo) const;

    //! Get a host track view
    GeoTrackView make_geo_track_view(TrackSlotId tsid = TrackSlotId{0});
    //! Get and initialize a single-thread host track view
    GeoTrackView make_geo_track_view(Real3 const& pos_cm, Real3 dir);

    //! Find linear segments until outside
    TrackingResult track(Real3 const& pos_cm, Real3 const& dir);
    //! Find linear segments until outside (maximum count
    TrackingResult track(Real3 const& pos_cm, Real3 const& dir, int max_step);

    //! Try to map Geant4 volumes using ImportVolume and name
    GeantVolResult
    get_import_geant_volumes(G4VPhysicalVolume const* world) const
    {
        return GeantVolResult::from_import(*this->geometry(), world);
    }
    //! Try to map Geant4 volumes using pointers
    GeantVolResult
    get_direct_geant_volumes(G4VPhysicalVolume const* world) const
    {
        return GeantVolResult::from_pointers(*this->geometry(), world);
    }

  private:
    template<Ownership W, MemSpace M>
    using StateData = typename TraitsT::template StateData<W, M>;
    using HostStateStore = CollectionStateStore<StateData, MemSpace::host>;

    SPConstGeo geo_;
    HostStateStore host_state_;

    SPConstGeoI build_fresh_geometry(std::string_view)
    {
        return this->build_geometry();
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
