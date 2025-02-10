//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Imported material data for optical models.
 *
 * Stores material properties relevant for Rayleigh scattering and
 * wavelength shifting.
 */
class ImportedMaterials
{
  public:
    // Construct from imported and shared data
    static std::shared_ptr<ImportedMaterials> from_import(ImportData const&);

    // Construct directly from imported materials
    ImportedMaterials(std::vector<ImportOpticalRayleigh> rayleigh,
                      std::vector<ImportWavelengthShift> wls);

    // Get number of imported optical materials
    OpticalMaterialId::size_type num_materials() const;

    // Get imported Rayleigh material parameters
    ImportOpticalRayleigh const& rayleigh(OpticalMaterialId mat) const;

    // Get imported wavelength shifting material parameters
    ImportWavelengthShift const& wls(OpticalMaterialId mat) const;

  private:
    std::vector<ImportOpticalRayleigh> rayleigh_;
    std::vector<ImportWavelengthShift> wls_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
