//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
// THE ABOVE TEXT APPLIES TO MODIFICATIONS FROM THE ORIGINAL WORK BELOW:
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
// Original work:
// https://gitlab.cern.ch/VecGeom/g4vecgeomnav/-/raw/fdd310842fa71c58b3d99646159ef1993a0366b0/include/G4VecGeomConverter.h
// Original code from G4VecGeomNav package by John Apostolakis et al.
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantGeoConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <utility>
#include <vector>
#include <VecGeom/base/TypeMap.h>
#include <VecGeom/management/GeoManager.h>

#include "celeritas/ext/VecgeomData.hh"

class G4LogicalVolume;
class G4AffineTransformation;
class G4VPhysicalVolume;
class G4VSolid;
class G4PVReplica;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert G4 geometry to VecGeom.
 */
class GeantGeoConverter
{
  private:
    //!@{
    //! \name Type aliases
    using Transformation3D = vecgeom::Transformation3D;
    using LogicalVolume = vecgeom::LogicalVolume;
    using VPlacedVolume = vecgeom::VPlacedVolume;
    using VUnplacedVolume = vecgeom::VUnplacedVolume;
    using VolumeId = celeritas::OpaqueId<struct Volume>;
    //!@}

  public:
    //! Default constructor.
    GeantGeoConverter() = default;

    //!@{
    //! Deleted constructors and assignment operator.
    GeantGeoConverter(GeantGeoConverter const&) = delete;
    GeantGeoConverter(GeantGeoConverter const&&) = delete;
    GeantGeoConverter& operator=(GeantGeoConverter const&) = delete;
    //!@}

    /*!
     * Main entry point of geometry importer.
     *
     * Queries the G4 geometry for the top volume and recursively
     * converts the whole native geometry into a VecGeom geometry.
     */
    VPlacedVolume const& operator()(G4VPhysicalVolume const*);

    /*!
     * Map between G4LogicalVolume* and volume IDs.
     */
    std::map<G4LogicalVolume const*, celeritas::VolumeId> get_g4logvol_id_map()
    {
        return std::move(g4logvol_id_map_);
    }

  private:
    //// TYPES ////

    using VecVPlacedVolume = std::vector<VPlacedVolume const*>;

    //// DATA ////

    // one G4 physical volume can correspond to multiple vecgeom placed volumes
    // (in case of replicas)
    std::map<G4VPhysicalVolume const*, VPlacedVolume const*> placed_volume_map_;

    std::map<G4VSolid const*, VUnplacedVolume const*> unplaced_volume_map_;

    std::map<G4LogicalVolume const*, LogicalVolume const*> logical_volume_map_;

    std::map<G4LogicalVolume const*, celeritas::VolumeId> g4logvol_id_map_;

    //// HELPER FUNCTIONS ////

    /*!
     * Convert a physical volume into a VecGeom placed volume.
     *
     * Its transformation matrix and its logical volume are also converted,
     * making the conversion process recursive, comprising the whole geometry
     * starting from the top volume.
     * Will take care not to convert anything twice by checking the
     * mapping between Geant4 and VecGeom geometry.
     */
    VPlacedVolume const*
    convert_physical(G4VPhysicalVolume const*,
                     LogicalVolume const* mother = nullptr);

    //! Special treatment needed for replicated volumes.
    void extract_replicated_transformations(
        G4PVReplica const&, std::vector<Transformation3D const*>&) const;

    //! Converts G4 solids into VecGeom unplaced volumes
    VUnplacedVolume* convert_solid(G4VSolid const*);

    /*!
     * Convert logical volumes from Geant4 into VecGeom.
     *
     * All daughters' physical volumes will be recursively converted.
     */
    LogicalVolume* convert_logical(G4LogicalVolume const*);

    // Update reflection map
    void fix_reflected(G4LogicalVolume const* refl_lv);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
