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
    //!@}

  public:
    //! Build the geometry
    virtual SPConstGeo build_geometry() = 0;

    // Access geometry
    SPConstGeo const& geometry();
    SPConstGeo const& geometry() const;

    // Get the name of the current volume
    std::string volume_name(GeoTrackView const& geo) const;
    // Get the name of the current surface if available
    std::string surface_name(GeoTrackView const& geo) const;

    // Get a single-thread host track view
    GeoTrackView make_geo_track_view();
    // Get and initialize a single-thread host track view
    GeoTrackView make_geo_track_view(Real3 const& pos, Real3 dir);

    // Calculate a "bumped" position based on the geo's state
    Real3 calc_bump_pos(GeoTrackView const& geo, real_type delta) const;

    //! Find linear segments until outside
    TrackingResult track(Real3 const& pos, Real3 const& dir);

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
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
using GenericCoreGeoTestBase = GenericVecgeomTestBase;
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
using GenericCoreGeoTestBase = GenericOrangeTestBase;
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
