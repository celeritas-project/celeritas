//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedMaterials.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/Types.hh"

namespace celeritas
{
struct ImportData;
struct ImportOpticalRayleigh;
struct ImportWavelengthShift;
class MaterialParams;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Imported material data for optical models.
 *
 * Stores material properties relevant for Rayleigh scattering and
 * wavelength shifting. A lookup table for optical to core material IDs is
 * also constructed, so models can quickly access core material properties.
 */
class ImportedMaterials
{
  public:
    //!@{
    //! \name Type aliases
    using CoreMaterialId = ::celeritas::MaterialId;
    using CoreMaterialParams = ::celeritas::MaterialParams;
    //!@}

  public:
    // Construct from imported and shared data
    static std::shared_ptr<ImportedMaterials>
    from_import(ImportData const&, CoreMaterialParams const&);

    // Construct directly from imported materials
    ImportedMaterials(std::vector<CoreMaterialId> core_material_map,
                      std::vector<ImportOpticalRayleigh> rayleigh,
                      std::vector<ImportWavelengthShift> wls);

    // Get number of imported optical materials
    OpticalMaterialId::size_type num_materials() const;

    // Get imported Rayleigh material parameters
    ImportOpticalRayleigh const& rayleigh(OpticalMaterialId mat) const;

    // Get imported wavelength shifting material parameters
    ImportWavelengthShift const& wls(OpticalMaterialId mat) const;

    // Get core material ID that corresponds to the optical material
    CoreMaterialId core_material_id(OpticalMaterialId mat) const;

  private:
    std::vector<CoreMaterialId> core_material_map_;
    std::vector<ImportOpticalRayleigh> rayleigh_;
    std::vector<ImportWavelengthShift> wls_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
