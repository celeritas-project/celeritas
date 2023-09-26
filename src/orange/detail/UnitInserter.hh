//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"
#include "orange/construct/OrangeInput.hh"

#include "BIHBuilder.hh"
#include "TransformInserter.hh"

namespace celeritas
{
namespace detail
{
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
    UnitInserter(Data* orange_data);

    // Create a simple unit and store in in OrangeParamsData
    SimpleUnitId operator()(UnitInput const& inp);

  private:
    Data* orange_data_{nullptr};
    BIHBuilder build_bih_tree_;
    TransformInserter insert_transform_;

    CollectionBuilder<SimpleUnitRecord> simple_units_;

    CollectionBuilder<LocalSurfaceId> local_surface_ids_;
    CollectionBuilder<LocalVolumeId> local_volume_ids_;
    CollectionBuilder<OpaqueId<real_type>> real_ids_;
    CollectionBuilder<logic_int> logic_ints_;
    CollectionBuilder<real_type> reals_;
    CollectionBuilder<SurfaceType> surface_types_;
    CollectionBuilder<ConnectivityRecord> connectivity_records_;
    CollectionBuilder<VolumeRecord> volume_records_;
    CollectionBuilder<Daughter> daughters_;

    //// HELPER METHODS ////

    SurfacesRecord insert_surfaces(SurfaceInput const& s);
    VolumeRecord
    insert_volume(SurfacesRecord const& unit, VolumeInput const& v);

    void process_daughter(VolumeRecord* vol_record,
                          DaughterInput const& daughter_input);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
