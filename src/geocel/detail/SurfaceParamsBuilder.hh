//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/SurfaceParamsBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/io/Label.hh"
#include "geocel/SurfaceData.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Temporary data structure to hold surface information for a volume.
 *
 * This is an intermediate representation of VolumeSurfaceRecord that uses
 * STL containers for easier construction before being converted to the
 * device-compatible format.
 */
struct VolumeSurfaceData
{
    //! Type for interface key (pre/post volume instance ID)
    using Interface = std::pair<VolumeInstanceId, VolumeInstanceId>;

    //! Surface identifier for the volume boundary (if any)
    SurfaceId boundary;

    //! Map of pre/post volume instance pairs to surface IDs
    std::map<Interface, SurfaceId> interfaces;

    //! True if any data is present
    explicit operator bool() const { return boundary || !interfaces.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Process input surfaces and build VolumeSurfaceData structures.
 *
 * This class processes inp::Surface data and constructs the necessary
 * VolumeSurfaceData objects that map surfaces to volumes and their interfaces.
 * It performs error checking against duplicate surface assignments.
 */
class SurfaceInputInserter
{
  public:
    // Construct with pointers to target data and volume params
    SurfaceInputInserter(VolumeParams const& volumes,
                         std::vector<Label>& labels,
                         std::vector<VolumeSurfaceData>& volume_surfaces);

    // Process an input surface and return its ID
    SurfaceId operator()(inp::Surface const& surf);

    // Process a boundary surface
    SurfaceId operator()(inp::Surface::Boundary const& boundary);

    // Process an interface surface
    SurfaceId operator()(inp::Surface::Interface const& interface);

  private:
    VolumeParams const& volumes_;
    std::vector<Label>& labels_;
    std::vector<VolumeSurfaceData>& volume_surfaces_;

    //// HELPER FUNCTIONS ////

    // Get the next surface ID to be added
    SurfaceId next_surface_id() const;
    // Get the label for a volume ID
    Label const& label(VolumeId vol_id) const;
    // Get the label for a volume instance ID
    Label const& label(VolumeInstanceId vol_inst_id) const;
    // Get the label for a surface ID
    Label const& label(SurfaceId surface_id) const;
};

//---------------------------------------------------------------------------//
/*!
 * Convert VolumeSurfaceData to VolumeSurfaceRecord for device storage.
 *
 * This class takes the temporary VolumeSurfaceData structure and converts it
 * to the device-compatible VolumeSurfaceRecord format, flattening the maps
 * into sorted arrays.
 */
class VolumeSurfaceRecordBuilder
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    template<class T>
    using VolumeItems
        = Collection<T, Ownership::value, MemSpace::host, VolumeId>;
    //!@}

    // Construct with pointers to target collections
    VolumeSurfaceRecordBuilder(VolumeItems<VolumeSurfaceRecord>* volume_surfaces,
                               Items<VolumeInstanceId>* volume_instance_ids,
                               Items<SurfaceId>* surface_ids);

    // Convert VolumeSurfaceData to VolumeSurfaceRecord
    VolumeId operator()(VolumeSurfaceData const& data);

  private:
    CollectionBuilder<VolumeSurfaceRecord, MemSpace::host, VolumeId>
        volume_surfaces_;
    DedupeCollectionBuilder<VolumeInstanceId> volume_instance_ids_;
    DedupeCollectionBuilder<SurfaceId> surface_ids_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
