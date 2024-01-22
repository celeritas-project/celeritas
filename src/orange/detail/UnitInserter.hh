//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"
#include "orange/construct/OrangeInput.hh"

#include "BIHBuilder.hh"
#include "SurfacesRecordBuilder.hh"
#include "TransformInserter.hh"

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
    UniverseId operator()(UnitInput const& inp);

  private:
    Data* orange_data_{nullptr};
    BIHBuilder build_bih_tree_;
    TransformInserter insert_transform_;
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
    CollectionBuilder<Daughter> daughters_;

    //// HELPER METHODS ////

    VolumeRecord
    insert_volume(SurfacesRecord const& unit, VolumeInput const& v);

    void process_daughter(VolumeRecord* vol_record,
                          DaughterInput const& daughter_input);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
