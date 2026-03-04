//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedMaterials.cc
//---------------------------------------------------------------------------//
#include "ImportedMaterials.hh"

#include <algorithm>
#include <vector>

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
    auto const& bulk_phys = data.optical_physics.bulk;
    if (!(bulk_phys.mie || bulk_phys.rayleigh || bulk_phys.wls
          || bulk_phys.wls2))
    {
        return nullptr;
    }

    auto assign_models = [](auto const& model_data, auto& out_vec) {
        for (auto const& [opt_mat_id, model_mat] : model_data.materials)
        {
            CELER_EXPECT(opt_mat_id < out_vec.size());
            out_vec[opt_mat_id.get()] = model_mat;
        }
    };

    OptMatId::size_type num_materials = data.optical_materials.size();
    std::vector<ImportMie> mie(num_materials);
    std::vector<ImportOpticalRayleigh> rayleigh(num_materials);
    std::vector<ImportWavelengthShift> wls(num_materials);
    std::vector<ImportWavelengthShift> wls2(num_materials);

    assign_models(bulk_phys.mie, mie);
    assign_models(bulk_phys.rayleigh, rayleigh);
    assign_models(bulk_phys.wls, wls);
    assign_models(bulk_phys.wls2, wls2);

    return std::make_shared<ImportedMaterials>(
        std::move(mie), std::move(rayleigh), std::move(wls), std::move(wls2));
}

//---------------------------------------------------------------------------//
/*!
 * Construct directly from imported material properties.
 */
ImportedMaterials::ImportedMaterials(std::vector<ImportMie> mie,
                                     std::vector<ImportOpticalRayleigh> rayleigh,
                                     std::vector<ImportWavelengthShift> wls,
                                     std::vector<ImportWavelengthShift> wls2)
    : mie_(std::move(mie))
    , rayleigh_(std::move(rayleigh))
    , wls_(std::move(wls))
    , wls2_(std::move(wls2))
{
    CELER_EXPECT(!rayleigh_.empty());
    CELER_EXPECT(rayleigh_.size() == wls_.size());
    CELER_EXPECT(rayleigh_.size() == wls2_.size());
    CELER_EXPECT(rayleigh_.size() == mie_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Number of imported optical materials.
 */
OptMatId::size_type ImportedMaterials::num_materials() const
{
    return rayleigh_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get imported Rayleigh properties for the given material.
 */
ImportOpticalRayleigh const& ImportedMaterials::rayleigh(OptMatId mat) const
{
    CELER_EXPECT(mat < this->num_materials());
    return rayleigh_[mat.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get imported wavelength shifting properties for the given material.
 */
ImportWavelengthShift const& ImportedMaterials::wls(OptMatId mat) const
{
    CELER_EXPECT(mat < this->num_materials());
    return wls_[mat.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get imported wavelength shifting properties for the given material.
 */
ImportWavelengthShift const& ImportedMaterials::wls2(OptMatId mat) const
{
    CELER_EXPECT(mat < this->num_materials());
    return wls2_[mat.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get imported Mie properties for the given material.
 */
ImportMie const& ImportedMaterials::mie(OptMatId mat) const
{
    CELER_EXPECT(mat < this->num_materials());
    return mie_[mat.get()];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
