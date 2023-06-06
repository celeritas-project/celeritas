//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GenericGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/geo/GeoFwd.hh"

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
    std::vector<real_type> distances;
    std::vector<real_type> halfway_safeties;

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
 * \tparam HP Geometry host Params class
 * \tparam S State data class
 * \tparam TV Track view clsas
 *
 * \note This class is instantiated in GenericGeoTestBase.cc for each available
 * geometry type.
 */
template<class HP, template<Ownership, MemSpace> class S, class TV>
class GenericGeoTestBase : virtual public Test, private LazyGeoManager
{
    static_assert(std::is_base_of_v<GeoParamsInterface, HP>);

  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<HP const>;
    using GeoTrackView = TV;
    using TrackingResult = GenericGeoTrackingResult;
    using GeantVolResult = GenericGeoGeantImportVolumeResult;
    //!@}

  public:
    //! Build the geometry
    virtual SPConstGeo build_geometry() = 0;

    // Access geometry
    SPConstGeo const& geometry();
    SPConstGeo const& geometry() const;

    //! Get the name of the current volume
    std::string volume_name(GeoTrackView const& geo) const;
    //! Get the name of the current surface if available
    std::string surface_name(GeoTrackView const& geo) const;

    //! Get a single-thread host track view
    GeoTrackView make_geo_track_view();
    //! Get and initialize a single-thread host track view
    GeoTrackView make_geo_track_view(Real3 const& pos, Real3 dir);

    //! Calculate a "bumped" position based on the geo's state
    Real3 calc_bump_pos(GeoTrackView const& geo, real_type delta) const;

    //! Find linear segments until outside
    TrackingResult track(Real3 const& pos, Real3 const& dir);
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
    using HostStateStore = CollectionStateStore<S, MemSpace::host>;

    SPConstGeo geo_;
    HostStateStore host_state_;

    SPConstGeoI build_fresh_geometry(std::string_view)
    {
        return this->build_geometry();
    }
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using GenericVecgeomTestBase
    = GenericGeoTestBase<VecgeomParams, VecgeomStateData, VecgeomTrackView>;
using GenericOrangeTestBase
    = GenericGeoTestBase<OrangeParams, OrangeStateData, OrangeTrackView>;
using GenericGeantGeoTestBase
    = GenericGeoTestBase<GeantGeoParams, GeantGeoStateData, GeantGeoTrackView>;
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
using GenericCoreGeoTestBase = GenericVecgeomTestBase;
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
using GenericCoreGeoTestBase = GenericOrangeTestBase;
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
using GenericCoreGeoTestBase = GenericGeantGeoTestBase;
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
