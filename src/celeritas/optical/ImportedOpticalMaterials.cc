//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedOpticalMaterials.cc
//---------------------------------------------------------------------------//
#include "ImportedOpticalMaterials.hh"

#include "celeritas/io/ImportData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from imported optical material data.
 */
std::shared_ptr<ImportedOpticalMaterials>
ImportedOpticalMaterials::from_import(ImportData const& data)
{
    std::vector<ImportOpticalMaterial> materials;

    for (auto const& material_pair : data.optical)
    {
        materials.push_back(material_pair.second);
    }

    return std::make_shared<ImportedOpticalMaterials>(std::move(materials));
}

//---------------------------------------------------------------------------//
/*!
 * Construct directly with imported optical materials.
 */
ImportedOpticalMaterials::ImportedOpticalMaterials(
    std::vector<ImportOpticalMaterial> io)
    : materials_(io)
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
