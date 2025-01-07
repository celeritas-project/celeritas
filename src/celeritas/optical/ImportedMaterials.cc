//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedMaterials.cc
//---------------------------------------------------------------------------//
#include "ImportedMaterials.hh"

#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/mat/MaterialParams.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct from imported and shared data.
 */
std::shared_ptr<ImportedMaterials>
ImportedMaterials::from_import(ImportData const& data,
                               CoreMaterialParams const& core_materials)
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

    // Construct optical -> core material map

    std::vector<CoreMaterialId> core_map(num_materials, CoreMaterialId{});
    for (auto core_id : range(CoreMaterialId{core_materials.num_materials()}))
    {
        if (auto opt_mat_id = core_materials.get(core_id).optical_material_id())
        {
            CELER_EXPECT(opt_mat_id < num_materials);
            core_map[opt_mat_id.get()] = core_id;
        }
    }

    CELER_ENSURE(
        std::all_of(core_map.begin(), core_map.end(), [](CoreMaterialId m) {
            return static_cast<bool>(m);
        }));

    return std::make_shared<ImportedMaterials>(
        std::move(core_map), std::move(rayleigh), std::move(wls));
}

//---------------------------------------------------------------------------//
/*!
 * Construct directly from imported material properties.
 */
ImportedMaterials::ImportedMaterials(
    std::vector<CoreMaterialId> core_material_map,
    std::vector<ImportOpticalRayleigh> rayleigh,
    std::vector<ImportWavelengthShift> wls)
    : core_material_map_(std::move(core_material_map))
    , rayleigh_(std::move(rayleigh))
    , wls_(std::move(wls))
{
    CELER_EXPECT(core_material_map_.size() == rayleigh_.size());
    CELER_EXPECT(core_material_map_.size() == wls_.size());
    CELER_EXPECT(std::all_of(
        core_material_map_.begin(), core_material_map_.end(), [](auto mat_id) {
            return static_cast<bool>(mat_id);
        }));
}

//---------------------------------------------------------------------------//
/*!
 * Number of imported optical materials.
 */
OpticalMaterialId::size_type ImportedMaterials::num_materials() const
{
    return core_material_map_.size();
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
/*!
 * Get the core material ID that maps to the given optical material ID.
 */
auto ImportedMaterials::core_material_id(OpticalMaterialId mat) const
    -> CoreMaterialId
{
    CELER_EXPECT(mat < this->num_materials());
    return core_material_map_[mat.get()];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
