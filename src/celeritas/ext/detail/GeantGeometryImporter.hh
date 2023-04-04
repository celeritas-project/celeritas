//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
// THE ABOVE TEXT APPLIES TO MODIFICATIONS FROM THE ORIGINAL WORK BELOW:
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
// Original work:
// https://gitlab.cern.ch/VecGeom/g4vecgeomnav/-/raw/fdd310842fa71c58b3d99646159ef1993a0366b0/include/G4VecGeomConverter.h
//---------------------------------------------------------------------------//
/*!
 * \file GeantGeometryImporter.hh
 * \brief Class to create a VecGeom model from a pre-existing Geant4 geometry
 *
 * Original code from G4VecGeomNav package by John Apostolakis et.al.
 */
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <utility>
#include <vector>
#include <G4PVReplica.hh>
#include <G4RotationMatrix.hh>
#include <G4ThreeVector.hh>
#include <VecGeom/base/TypeMap.h>
#include <VecGeom/management/GeoManager.h>

class G4LogicalVolume;
class G4AffineTransformation;
class G4VPhysicalVolume;
class G4VSolid;

namespace celeritas
{

//! class converting G4 to VecGeom (only geometry; no materials)
class GeantGeometryImporter
{
  private:
    //!@{
    //! \name Type aliases
    using Transformation3D = vecgeom::Transformation3D;
    using LogicalVolume = vecgeom::LogicalVolume;
    using VPlacedVolume = vecgeom::VPlacedVolume;
    using VUnplacedVolume = vecgeom::VUnplacedVolume;
    //!@}

  public:
    //! Default constructor.
    GeantGeometryImporter() = default;

    /*!
     * Main entry point of geometry importer.
     *
     * Queries the G4 geometry for the top volume and recursively
     * converts the whole native geometry into a VecGeom geometry.
     */
    VPlacedVolume const& operator()(G4VPhysicalVolume const*);

    //! Returns a placed volume that corresponds to a G4VPhysicalVolume.
    std::vector<VPlacedVolume const*> const*
    get_placed_volume(G4VPhysicalVolume const* n) const
    {
        if (n == nullptr)
            return nullptr;
        if (auto found = placed_volume_map_.find(n);
            found != placed_volume_map_.end())
            return found->second;
        else
            return nullptr;
    }

  private:
    /*!
     * Converts a physical volume into a VecGeom placed volume.
     *
     * Its transformation matrix and its logical volume are also converted,
     * making the conversion process recursive, comprising the whole geometry
     * starting from the top volume.
     * Will take care not to convert anything twice by checking the
     * mapping between Geant4 and VecGeom geometry.
     */
    std::vector<VPlacedVolume const*> const* convert(G4VPhysicalVolume const*);

    /**
     * @brief Special treatment needed for replicated volumes.
     */
    void extract_replicated_transformations(
        G4PVReplica const&, std::vector<Transformation3D const*>&) const;

    //! Converts G4 solids into VecGeom unplaced volumes
    VUnplacedVolume* convert(G4VSolid const*);

    /*! Converts logical volumes from Geant4 into VecGeom.
     *
     * All daughters' physical volumes will be recursively converted.
     */
    LogicalVolume* convert(G4LogicalVolume const*);

    //! Convert transformation matrices.
    Transformation3D* convert(G4ThreeVector const&, G4RotationMatrix const*);

  private:
    //!@{ Deleted constructors and assignment operator.
    GeantGeometryImporter(GeantGeometryImporter const&) = delete;
    GeantGeometryImporter(GeantGeometryImporter const&&) = delete;
    GeantGeometryImporter& operator=(GeantGeometryImporter const&) = delete;
    //!@}

  private:
    //! Pointer to generated world built from imported Geant4 geometry.
    VPlacedVolume const* world_;

    // one G4 physical volume can correspond to multiple vecgeom placed volumes
    // (in case of replicas)
    std::map<G4VPhysicalVolume const*, std::vector<VPlacedVolume const*> const*>
        placed_volume_map_;

    std::map<G4VSolid const*, VUnplacedVolume const*> unplaced_volume_map_;

    std::map<G4LogicalVolume const*, LogicalVolume const*> logical_volume_map_;

    std::vector<Transformation3D const*> replica_transformations_;
};
}  // namespace celeritas
