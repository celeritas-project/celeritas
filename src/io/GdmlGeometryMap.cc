//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GdmlGeometryMap.cc
//---------------------------------------------------------------------------//
#include "GdmlGeometryMap.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct/destruct with defaults
 */
GdmlGeometryMap::GdmlGeometryMap()  = default;
GdmlGeometryMap::~GdmlGeometryMap() = default;

//---------------------------------------------------------------------------//
// READ
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Return the mat_id for a given vol_id
 */
mat_id GdmlGeometryMap::get_matid(vol_id volume_id) const
{
    auto iter = volid_to_matid_.find(volume_id);
    CELER_EXPECT(iter != volid_to_matid_.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Return the ImportVolume associated with the vol_id
 */
const ImportVolume& GdmlGeometryMap::get_volume(vol_id volume_id) const
{
    auto iter = volid_to_volume_.find(volume_id);
    CELER_EXPECT(iter != volid_to_volume_.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Return the ImportMaterial associated with the mat_id
 */
const ImportMaterial& GdmlGeometryMap::get_material(mat_id material_id) const
{
    auto iter = matid_to_material_.find(material_id);
    CELER_EXPECT(iter != matid_to_material_.end());
    return iter->second;
}
//---------------------------------------------------------------------------//
/*!
 * Return the ImportElement associated with the elem_id
 */
const ImportElement& GdmlGeometryMap::get_element(elem_id element_id) const
{
    auto iter = elemid_to_element_.find(element_id);
    CELER_EXPECT(iter != elemid_to_element_.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Return the size of the largest material element list
 *
 * Can be used to preallocate storage for each thread for XS calculations
 */
size_type GdmlGeometryMap::max_num_elements() const
{
    size_type result = 0;
    for (const auto& key : matid_to_material_)
    {
        result = std::min(result, key.second.elements_fractions.size());
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return a reference to private member matid_to_material_
 */
const std::map<mat_id, ImportMaterial>&
GdmlGeometryMap::matid_to_material_map() const
{
    return matid_to_material_;
}

//---------------------------------------------------------------------------//
/*!
 * Return a reference to private member volid_to_volume_
 */
const std::map<vol_id, ImportVolume>&
GdmlGeometryMap::volid_to_volume_map() const
{
    return volid_to_volume_;
}

//---------------------------------------------------------------------------//
/*!
 * Return a reference to private member elemid_to_element_
 */
const std::map<elem_id, ImportElement>&
GdmlGeometryMap::elemid_to_element_map() const
{
    return elemid_to_element_;
}

//---------------------------------------------------------------------------//
/*!
 * Return a reference to private member volid_to_matid_
 */
const std::map<vol_id, mat_id>& GdmlGeometryMap::volid_to_matid_map() const
{
    return volid_to_matid_;
}

//---------------------------------------------------------------------------//
// WRITE
//---------------------------------------------------------------------------//
/*!
 * Add pair <mat_id, ImportMaterial> to the matid_to_material_ map
 */
void GdmlGeometryMap::add_material(mat_id id, const ImportMaterial& material)
{
    auto result = matid_to_material_.insert({id, material});
    CELER_ASSERT(result.second);
}

//---------------------------------------------------------------------------//
/*!
 * Add pair <vol_id, volume> to the volid_to_volume_ map
 */
void GdmlGeometryMap::add_volume(vol_id id, const ImportVolume& volume)
{
    volid_to_volume_.insert({id, volume});
}
//---------------------------------------------------------------------------//
/*!
 * Add pair <elem_id, element> to the elemid_to_element_ map
 */
void GdmlGeometryMap::add_element(elem_id id, const ImportElement& element)
{
    elemid_to_element_.insert({id, element});
}

//---------------------------------------------------------------------------//
/*!
 * Add pair <vol_id, mat_id> to the volid_to_matid_ map.
 * This links geometry and material information
 */
void GdmlGeometryMap::link_volume_material(vol_id volid, mat_id matid)
{
    volid_to_matid_.insert({volid, matid});
}

//---------------------------------------------------------------------------//
} // namespace celeritas
