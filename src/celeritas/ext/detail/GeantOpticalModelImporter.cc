//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantOpticalModelImporter.cc
//---------------------------------------------------------------------------//
#include "GeantOpticalModelImporter.hh"

#include <G4Material.hh>
#include <G4MaterialPropertiesTable.hh>

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/io/ImportMaterial.hh"

#include "GeantMaterialPropertyGetter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model importer.
 *
 * The vector of materials is used to create an optical material ID to
 * G4Material map. Unique materials should have unique optical material IDs.
 */
GeantOpticalModelImporter::GeantOpticalModelImporter(
    std::vector<ImportPhysMaterial> const& materials)
{
    using Index = OpaqueId<G4Material, typename ImportPhysMaterial::Index>;

    // Create optical material -> G4MaterialPropertiesTable lookup
    Index::size_type num_opt_materials = 0;
    for (auto const& mat : materials)
    {
        if (Index{mat.optical_material_id})
        {
            num_opt_materials
                = std::max(num_opt_materials, mat.optical_material_id + 1);
        }
    }
    opt_to_mat_ = std::vector<G4MaterialPropertiesTable const*>(
        num_opt_materials, nullptr);

    auto const& mt = *G4Material::GetMaterialTable();
    CELER_ASSERT(mt.size() == materials.size());

    for (auto mat_idx : range(mt.size()))
    {
        if (auto opt_id = Index{materials[mat_idx].optical_material_id})
        {
            G4Material const* material = mt[mat_idx];
            CELER_ASSERT(material);

            auto& mapped_mpt = opt_to_mat_[opt_id.get()];
            auto const* mpt = material->GetMaterialPropertiesTable();

            // Different material properties shouldn't map to the same optical
            // ID
            CELER_EXPECT(!mapped_mpt || mapped_mpt == mpt);

            mapped_mpt = mpt;

            // Optical IDs should have material property tables already
            // associated with them
            CELER_ASSERT(mapped_mpt);
        }
    }

    CELER_ASSERT(std::all_of(
        opt_to_mat_.begin(), opt_to_mat_.end(), [](auto const* mpt) {
            return static_cast<bool>(mpt);
        }));
}

//---------------------------------------------------------------------------//
/*!
 * Create ImportOpticalMaterial for the given model class.
 *
 * If the model class does not correspond to a supported MFP property name,
 * then an empty ImportOpticalModel with model class size_ is returned.
 */
ImportOpticalModel GeantOpticalModelImporter::operator()(IMC imc) const
{
    switch (imc)
    {
        case IMC::absorption:
            return ImportOpticalModel{imc, this->import_mfps("ABSLENGTH")};
        case IMC::rayleigh:
            return ImportOpticalModel{imc, this->import_mfps("RAYLEIGH")};
        case IMC::wls:
            return ImportOpticalModel{imc, this->import_mfps("WLSABSLENGTH")};
        default:
            return ImportOpticalModel{IMC::size_, {}};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Import MFP table with the given property name.
 */
std::vector<ImportPhysicsVector>
GeantOpticalModelImporter::import_mfps(std::string const& mfp_property_name) const
{
    std::vector<ImportPhysicsVector> mfps(opt_to_mat_.size());
    for (auto opt_idx : range(mfps.size()))
    {
        GeantMaterialPropertyGetter get_property{*opt_to_mat_[opt_idx]};
        get_property(&mfps[opt_idx], mfp_property_name, ImportUnits::len);
    }
    return mfps;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
