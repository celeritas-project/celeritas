//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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

//---------------------------------------------------------------------------//
/*!
 * \def CELERTEST_INST_GEO
 *
 * Explicitly instantiate a class for each available geometry params type.
 *
 * In a .cc file:
 * \code
 * CELERTEST_INST_GEO(MyTypedTestClass);
 * \endcode
 */
//---------------------------------------------------------------------------//

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

namespace testdetail
{
//---------------------------------------------------------------------------//
template<class HP>
struct GenericGeoTraits;

template<>
struct GenericGeoTraits<VecgeomParams>
{
    template<MemSpace M>
    using StateStore = CollectionStateStore<VecgeomStateData, M>;
    using TrackView = VecgeomTrackView;
    static inline char const* ext = ".gdml";
    static inline char const* name = "VecGeom";
};

template<>
struct GenericGeoTraits<OrangeParams>
{
    template<MemSpace M>
    using StateStore = CollectionStateStore<OrangeStateData, M>;

    using TrackView = OrangeTrackView;
    static inline char const* ext = ".org.json";
    static inline char const* name = "ORANGE";
};

template<>
struct GenericGeoTraits<GeantGeoParams>
{
    template<MemSpace M>
    using StateStore = CollectionStateStore<GeantGeoStateData, M>;
    using TrackView = GeantGeoTrackView;
    static inline char const* ext = ".gdml";
    static inline char const* name = "Geant4";
};

//---------------------------------------------------------------------------//
}  // namespace testdetail

//---------------------------------------------------------------------------//
/*!
 * Templated base class for loading geometry.
 *
 * \tparam HP Geometry host Params class
 *
 * \sa AllGeoTypedTestBase
 *
 * \note This class is instantiated in GenericGeoTestBase.cc for each available
 * geometry type.
 */
template<class HP>
class GenericGeoTestBase : virtual public Test, private LazyGeoManager
{
    static_assert(std::is_base_of_v<GeoParamsInterface, HP>);

    using TraitsT = testdetail::GenericGeoTraits<HP>;

  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<HP const>;
    using GeoTrackView = typename TraitsT::TrackView;
    using TrackingResult = GenericGeoTrackingResult;
    using GeantVolResult = GenericGeoGeantImportVolumeResult;
    //!@}

  public:
    //! Get the basename or unique geometry key (defaults to suite name)
    virtual std::string geometry_basename() const;

    //! Build the geometry
    virtual SPConstGeo build_geometry() = 0;

    //! Construct from celeritas test data and "basename" value
    SPConstGeo build_geometry_from_basename();

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
    using HostStateStore =
        typename TraitsT::template StateStore<MemSpace::host>;

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

using GenericVecgeomTestBase = GenericGeoTestBase<VecgeomParams>;
using GenericOrangeTestBase = GenericGeoTestBase<OrangeParams>;
using GenericGeantGeoTestBase = GenericGeoTestBase<GeantGeoParams>;

using GenericCoreGeoTestBase = GenericGeoTestBase<GeoParams>;

#define CELERTEST_INST_IMPL_(CLS, GEO) template class CLS<GEO>

#if CELERITAS_USE_VECGEOM
#    define CELERTEST_INST_VG_(CLS) CELERTEST_INST_IMPL_(CLS, VecgeomParams);
#else
#    define CELERTEST_INST_VG_(CLS)
#endif
#if CELERITAS_USE_GEANT4 && CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
#    define CELERTEST_INST_G4_(CLS) CELERTEST_INST_IMPL_(CLS, GeantGeoParams);
#else
#    define CELERTEST_INST_G4_(CLS)
#endif

// See documentation at top of file
#define CELERTEST_INST_GEO(CLS) \
    CELERTEST_INST_VG_(CLS)     \
    CELERTEST_INST_G4_(CLS)     \
    CELERTEST_INST_IMPL_(CLS, OrangeParams)

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
