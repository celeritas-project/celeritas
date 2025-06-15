//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantOpticalModelImporter.cc
//---------------------------------------------------------------------------//
#include "GeantOpticalModelImporter.hh"

#include <G4Material.hh>
#include <G4MaterialPropertiesTable.hh>

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
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
    GeoOpticalIdMap const& geo_to_opt)
    : opt_to_mat_{geo_to_opt.num_optical(), nullptr}
{
    if (geo_to_opt.empty())
    {
        return;
    }

    auto const& mt = *G4Material::GetMaterialTable();
    for (auto geo_mat_id : range(GeoMatId(mt.size())))
    {
        auto opt_id = geo_to_opt[geo_mat_id];
        if (!opt_id)
        {
            continue;
        }

        // Save properties tables
        G4Material const* material = mt[geo_mat_id.get()];
        CELER_ASSERT(material);
        opt_to_mat_[opt_id.get()] = material->GetMaterialPropertiesTable();
    }

    CELER_ASSERT(
        std::all_of(opt_to_mat_.begin(), opt_to_mat_.end(), LogicalTrue{}));
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
    CELER_EXPECT(*this);
    switch (imc)
    {
        case IMC::absorption:
            return ImportOpticalModel{imc, this->import_mfps("ABSLENGTH")};
        case IMC::rayleigh:
            return ImportOpticalModel{imc, this->import_mfps("RAYLEIGH")};
        case IMC::wls:
            return ImportOpticalModel{imc, this->import_mfps("WLSABSLENGTH")};
        case IMC::wls2:
            return ImportOpticalModel{imc, this->import_mfps("WLSABSLENGTH2")};
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Import MFP table with the given property name.
 */
std::vector<inp::Grid>
GeantOpticalModelImporter::import_mfps(std::string const& mfp_property_name) const
{
    std::vector<inp::Grid> mfps(opt_to_mat_.size());
    for (auto opt_idx : range(mfps.size()))
    {
        GeantMaterialPropertyGetter get_property{*opt_to_mat_[opt_idx]};
        get_property(&mfps[opt_idx],
                     mfp_property_name,
                     {ImportUnits::mev, ImportUnits::len});
    }
    return mfps;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
