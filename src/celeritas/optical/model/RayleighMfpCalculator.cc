//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighMfpCalculator.cc
//---------------------------------------------------------------------------//
#include "RayleighMfpCalculator.hh"

#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/mat/MaterialParams.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct Rayleigh material data from imported data and core Material
 * parameters.
 */
std::vector<OpticalRayleighMaterial> OpticalRayleighMaterial::from_import(
    std::vector<ImportOpticalRayleigh> const& imported,
    ::celeritas::MaterialParams const& mat)
{
    // Copy over imported data
    std::vector<OpticalRayleighMaterial> rayleigh;
    rayleigh.reserve(imported.size());
    for (ImportOpticalRayleigh const& import_rayl : imported)
    {
        OpticalRayleighMaterial rayl;
        rayl.scale_factor = import_rayl.scale_factor;
        rayl.compressibility = import_rayl.compressibility;
        rayleigh.push_back(rayl);
    }
    CELER_ENSURE(rayleigh.size() == imported.size());

    // Copy material temperatures
    for (auto mat_id : range(MaterialId{mat.num_materials()}))
    {
        auto mat_view = mat.get(mat_id);
        if (OpticalMaterialId opt_mat = mat_view.optical_material_id())
        {
            CELER_VALIDATE(opt_mat < rayleigh.size(),
                           << "mismatch between number of optical materials "
                              "and number of imported optical Rayleigh "
                              "properties");

            rayleigh[opt_mat.get()].temperature = mat_view.temperature();
        }
    }

    return rayleigh;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
