//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"

#include "BIHBuilder.hh"
#include "SurfacesRecordBuilder.hh"
#include "TransformRecordInserter.hh"
#include "../BoundingBoxUtils.hh"
#include "../OrangeData.hh"
#include "../OrangeInput.hh"
#include "../OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
class UniverseInserter;
//---------------------------------------------------------------------------//
/*!
 * Convert a unit input to params data.
 *
 * Linearize the data in a UnitInput and add it to the host.
 */
class UnitInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<OrangeParamsData>;
    //!@}

  public:
    // Construct from full parameter data
    UnitInserter(UniverseInserter* insert_universe, Data* orange_data);

    // Create a simple unit and store in in OrangeParamsData
    UnivId operator()(UnitInput&& inp);

  private:
    Data* orange_data_{nullptr};
    BIHBuilder build_bih_tree_;
    TransformRecordInserter insert_transform_;
    SurfacesRecordBuilder build_surfaces_;
    UniverseInserter* insert_universe_;

    CollectionBuilder<SimpleUnitRecord> simple_units_;

    DedupeCollectionBuilder<LocalSurfaceId> local_surface_ids_;
    DedupeCollectionBuilder<LocalVolumeId> local_volume_ids_;
    DedupeCollectionBuilder<OpaqueId<real_type>> real_ids_;
    DedupeCollectionBuilder<logic_int> logic_ints_;
    DedupeCollectionBuilder<real_type> reals_;
    DedupeCollectionBuilder<SurfaceType> surface_types_;
    CollectionBuilder<ConnectivityRecord> connectivity_records_;
    CollectionBuilder<VolumeRecord> volume_records_;
    CollectionBuilder<OrientedBoundingZoneRecord> obz_records_;
    CollectionBuilder<Daughter> daughters_;
    BoundingBoxBumper<fast_real_type, real_type> calc_bumped_;

    //// HELPER METHODS ////

    VolumeRecord
    insert_volume(SurfacesRecord const& unit, VolumeInput const& v);

    void process_daughter(VolumeRecord* vol_record,
                          DaughterInput const& daughter_input);

    void process_obz_record(VolumeRecord* vol_record,
                            OrientedBoundingZoneInput const& obz_input);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
