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
const std::string& MaterialParams::id_to_label(ElementDefId el) const
{
    REQUIRE(el < elnames_.size());
    return elnames_[el.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for an material.
 */
const std::string& MaterialParams::id_to_label(MaterialDefId mat) const
{
    REQUIRE(mat < matnames_.size());
    return matnames_[mat.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Find the material ID corresponding to a name.
 */
MaterialDefId MaterialParams::find(const std::string& name) const
{
    auto iter = matname_to_id_.find(name);
    if (iter == matname_to_id_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
