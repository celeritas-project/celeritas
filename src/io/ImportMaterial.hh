//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportMaterial.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "ImportElement.hh"
#include "GdmlGeometryMapTypes.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enum for storing G4State enumerators.
 * [See G4Material.hh]
 */
enum class ImportMaterialState
{
    not_defined,
    solid,
    liquid,
    gas
};

enum class ImportProductionCut
{
    gamma,
    electron,
    positron,
    proton,
    size
};

//---------------------------------------------------------------------------//
/*!
 * Store data of a given material and its elements.
 *
 * Used by the GdmlGeometryMap class.
 *
 * The data is exported via the app/geant-exporter. For further expanding
 * this struct, add the appropriate variables here and fetch the new values in
 * \c app/geant-exporter.cc:store_geometry(...).
 *
 * Units are defined at export time in the aforementioned function.
 */
struct ImportMaterial
{
    std::string         name;
    ImportMaterialState state;
    real_type           temperature;                                // [K]
    real_type           density;                                    // [g/cm^3]
    real_type           electron_density;                           // [1/cm^3]
    real_type           number_density;                             // [1/cm^3]
    real_type           radiation_length;                           // [cm]
    real_type           nuclear_int_length;                         // [cm]
    real_type           range_cuts[(int)ImportProductionCut::size]; // [cm]
    real_type           energy_cuts[(int)ImportProductionCut::size]; // [MeV]
    std::map<elem_id, real_type> elements_fractions;     // Mass fractions
    std::map<elem_id, real_type> elements_num_fractions; // Number fractions
};

//---------------------------------------------------------------------------//
} // namespace celeritas
