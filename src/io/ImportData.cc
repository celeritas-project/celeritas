//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportData.cc
//---------------------------------------------------------------------------//
#include "ImportData.hh"

#include <algorithm>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find an \c ImportElement object from \c ImportData using \c element_id .

 * This simplifies the process of fetching full elemental information from a
 * given material, as \c ImportMatElemComponent only stores the \c element_id .
 */
const ImportElement
get_import_element(const ImportData& data, unsigned int element_id)
{
    CELER_EXPECT(element_id <= data.elements.size());

    auto iter = std::find_if(data.elements.begin(),
                             data.elements.end(),
                             [element_id](const ImportElement& element) {
                                 return element.element_id == element_id;
                             });

    CELER_ENSURE(iter != data.elements.end());
    return *iter;
}

//---------------------------------------------------------------------------//
/*!
 * Find an \c ImportMaterial object from \c ImportData using \c material_id .

 * This simplifies the process of fetching full material information from a
 * given volume, as \c ImportVolume only stores the \c material_id .
 */
const ImportMaterial
get_import_material(const ImportData& data, unsigned int material_id)
{
    CELER_EXPECT(material_id <= data.materials.size());

    auto iter = std::find_if(data.materials.begin(),
                             data.materials.end(),
                             [material_id](const ImportMaterial& material) {
                                 return material.material_id == material_id;
                             });

    CELER_ENSURE(iter != data.materials.end());
    return *iter;
}
//---------------------------------------------------------------------------//
} // namespace celeritas
