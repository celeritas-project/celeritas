//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GdmlGeometryMap.hh
//! Store and link volume with material information.
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <map>

#include "ImportMaterial.hh"
#include "ImportVolume.hh"
#include "GdmlGeometryMapTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store material, element, and volume information.
 *
 * - The material id maps materials in the global material map. It also
 *   represents the position of the material in the vector<ImportPhysicsVector>
 *   of the ImportPhysicsTable.
 * - The element id maps elements in the global element map.
 * - The volume id maps volumes in the global volume map.
 * - Volume id and material id are linked by a map, so that given a volume id
 *   one can retrieve full material/element information, including XS data.
 */
class GdmlGeometryMap
{
  public:
    // Construct/destruct with defaults
    GdmlGeometryMap();
    ~GdmlGeometryMap();

    /// READ ///

    // Find material id given volume id
    mat_id get_matid(vol_id volume_id) const;
    // Find ImportVolume given volume id
    const ImportVolume& get_volume(vol_id volume_id) const;
    // Find ImportMaterial given a material id
    const ImportMaterial& get_material(mat_id material_id) const;
    // Find ImportElement given element id
    const ImportElement& get_element(elem_id element_id) const;

    // Return a reference to the private member volid_to_matid_ map
    const std::map<vol_id, mat_id>& volid_to_matid_map();

    /// WRITE (only used by geant-exporter app) ///

    // Add pair <mat_id, material> to the map
    void add_material(mat_id id, const ImportMaterial& material);
    // Add pair <vol_id, volume> to the map
    void add_volume(vol_id id, const ImportVolume& volume);
    // Add pair <elem_id, element> to the map
    void add_element(elem_id id, const ImportElement& element);
    // Add pair <vol_id, mat_id> to the map
    void link_volume_material(vol_id volid, mat_id matid);

  private:
    // Global maps
    std::map<mat_id, ImportMaterial> matid_to_material_;
    std::map<vol_id, ImportVolume>   volid_to_volume_;
    std::map<elem_id, ImportElement> elemid_to_element_;
    // Link between volume and material
    std::map<vol_id, mat_id> volid_to_matid_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
