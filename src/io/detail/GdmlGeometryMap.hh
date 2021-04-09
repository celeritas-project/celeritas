//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GdmlGeometryMap.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <string>

#include "ImportMaterial.hh"
#include "ImportVolume.hh"
#include "GdmlGeometryMapTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store material, element, and volume information.
 *
 * - The \c mat_id maps materials in the global material map. It also
 *   represents the position of said material in the \c ImportPhysicsTable
 *   vectors: \c ImportPhysicsTable.physics_vectors.at(mat_id) .
 * - The \c elem_id maps elements in the global element map.
 * - The \c vol_id maps volumes in the global volume map.
 * - \c vol_id and \c mat_id pairs are also mapped, such that from a \c vol_id
 *   one can fully retrieve all material and element information.
 * 
 * This data is exported via the \e geant-exporter in
 * \c geant-exporter.cc:store_geometry(...) .
 *
 * \sa ImportData
 */
class GdmlGeometryMap
{
  public:
    //!@{
    //! Type aliases
    using size_type = std::size_t;
    //!@}

  public:
    //// READ ////

    // Find material id given volume id
    mat_id get_matid(vol_id volume_id) const;
    // Find ImportVolume given volume id
    const ImportVolume& get_volume(vol_id volume_id) const;
    // Find ImportMaterial given a material id
    const ImportMaterial& get_material(mat_id material_id) const;
    // Find ImportElement given element id
    const ImportElement& get_element(elem_id element_id) const;

    // Return the size of the largest material element list
    auto max_num_elements() const -> size_type;

    // Return a reference to matid_to_material map
    const std::map<mat_id, ImportMaterial>& matid_to_material_map() const;
    // Return a reference to volid_to_volume_ map
    const std::map<vol_id, ImportVolume>& volid_to_volume_map() const;
    // Return a reference to elemid_to_element_ map
    const std::map<elem_id, ImportElement>& elemid_to_element_map() const;
    // Return a reference to volid_to_matid_ map
    const std::map<vol_id, mat_id>& volid_to_matid_map() const;

    //// WRITE (only used by geant-exporter app) ////

    // Add pair <mat_id, material> to the map
    void add_material(mat_id id, const ImportMaterial& material);
    // Add pair <vol_id, volume> to the map
    void add_volume(vol_id id, const ImportVolume& volume);
    // Add pair <elem_id, element> to the map
    void add_element(elem_id id, const ImportElement& element);
    // Add pair <vol_id, mat_id> to the map
    void link_volume_material(vol_id volid, mat_id matid);

    // Boolean operator for assertion macros
    explicit operator bool() const
    {
        return !matid_to_material_.empty() && !volid_to_volume_.empty()
               && !elemid_to_element_.empty() && !volid_to_matid_.empty();
    }

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
