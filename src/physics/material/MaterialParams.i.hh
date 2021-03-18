//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialParams.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the label for an element.
 */
const std::string& MaterialParams::id_to_label(ElementId el) const
{
    CELER_EXPECT(el < elnames_.size());
    return elnames_[el.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a material.
 */
const std::string& MaterialParams::id_to_label(MaterialId mat) const
{
    CELER_EXPECT(mat < matnames_.size());
    return matnames_[mat.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Find the material ID corresponding to a name.
 *
 * This function will have to be updated to allow for multiple MaterialDefIds
 * with same material name.
 */
MaterialId MaterialParams::find(const std::string& name) const
{
    auto iter = matname_to_id_.find(name);
    if (iter == matname_to_id_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Get material properties for the given material.
 */
MaterialView MaterialParams::get(MaterialId id) const
{
    CELER_EXPECT(id < this->host_pointers().materials.size());
    return MaterialView(this->host_pointers(), id);
}

//---------------------------------------------------------------------------//
/*!
 * Get properties for the given element.
 */
ElementView MaterialParams::get(ElementId id) const
{
    CELER_EXPECT(id < this->host_pointers().elements.size());
    return ElementView(this->host_pointers(), id);
}

//---------------------------------------------------------------------------//
/*!
 * Maximum number of elements in any one material.
 */
ElementComponentId::size_type MaterialParams::max_element_components() const
{
    return this->host_pointers().max_element_components;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
