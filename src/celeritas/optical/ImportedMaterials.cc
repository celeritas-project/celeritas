//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedMaterials.cc
//---------------------------------------------------------------------------//
#include "ImportedMaterials.hh"

#include <algorithm>

#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct from imported and shared data.
 */
std::shared_ptr<ImportedMaterials>
ImportedMaterials::from_import(ImportData const& data)
{
    // If there's no material specific parameters, return a nullptr
    if (!std::any_of(data.optical_materials.begin(),
                     data.optical_materials.end(),
                     [](auto const& mat) { return mat.rayleigh || mat.wls; }))
    {
        return nullptr;
    }

    OpticalMaterialId::size_type num_materials = data.optical_materials.size();

    // Copy over Rayleigh and WLS data

    std::vector<ImportOpticalRayleigh> rayleigh;
    rayleigh.reserve(num_materials);

    std::vector<ImportWavelengthShift> wls;
    wls.reserve(num_materials);

    for (auto const& mat : data.optical_materials)
    {
        rayleigh.push_back(mat.rayleigh);
        wls.push_back(mat.wls);
    }

    CELER_ENSURE(rayleigh.size() == num_materials);
    CELER_ENSURE(wls.size() == num_materials);

    return std::make_shared<ImportedMaterials>(std::move(rayleigh),
                                               std::move(wls));
}

//---------------------------------------------------------------------------//
/*!
 * Construct directly from imported material properties.
 */
ImportedMaterials::ImportedMaterials(std::vector<ImportOpticalRayleigh> rayleigh,
                                     std::vector<ImportWavelengthShift> wls)
    : rayleigh_(std::move(rayleigh)), wls_(std::move(wls))
{
    CELER_EXPECT(!rayleigh_.empty());
    CELER_EXPECT(rayleigh_.size() == wls_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Number of imported optical materials.
 */
OpticalMaterialId::size_type ImportedMaterials::num_materials() const
{
    return rayleigh_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get imported Rayleigh properties for the given material.
 */
ImportOpticalRayleigh const&
ImportedMaterials::rayleigh(OpticalMaterialId mat) const
{
    CELER_EXPECT(mat < this->num_materials());
    return rayleigh_[mat.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get imported wavelength shifting properties for the given material.
 */
ImportWavelengthShift const& ImportedMaterials::wls(OpticalMaterialId mat) const
{
    CELER_EXPECT(mat < this->num_materials());
    return wls_[mat.get()];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
