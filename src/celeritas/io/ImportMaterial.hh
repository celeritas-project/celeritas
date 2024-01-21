//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportMaterial.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <string>
#include <vector>

#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enum for storing G4State enumerators.
 */
enum class ImportMaterialState
{
    other,
    solid,
    liquid,
    gas,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Scalar optical properties.
 */
enum class ImportOpticalScalar
{
    resolution_scale,  //!< Scintillation
    rise_time_fast,
    rise_time_mid,
    rise_time_slow,
    fall_time_fast,
    fall_time_mid,
    fall_time_slow,
    scint_yield,
    scint_yield_fast,
    scint_yield_mid,
    scint_yield_slow,
    lambda_mean_fast,
    lambda_mean_mid,
    lambda_mean_slow,
    lambda_sigma_fast,
    lambda_sigma_mid,
    lambda_sigma_slow,
    size_  //!< Sentinel value
};

//---------------------------------------------------------------------------//
/*!
 * Vector optical properties.
 */
enum class ImportOpticalVector
{
    refractive_index,  //!< Common properties
    size_  //!< Sentinel value
};

//---------------------------------------------------------------------------//
/*!
 * Store particle production cut.
 */
struct ImportProductionCut
{
    double energy{};  //!< [MeV]
    double range{};  //!< [length]
};

//---------------------------------------------------------------------------//
/*!
 * Store elemental composition of a given material.
 */
struct ImportMatElemComponent
{
    unsigned int element_id{};  //!< Index of element in ImportElement
    double number_fraction{};  //!< [unitless]
};

//---------------------------------------------------------------------------//
/*!
 * Store optical material properties.
 */
struct ImportOpticalMaterial
{
    std::map<ImportOpticalScalar, double> scalars;
    std::map<ImportOpticalVector, ImportPhysicsVector> vectors;

    explicit operator bool() const
    {
        return !scalars.empty() || !vectors.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store material data.
 */
struct ImportMaterial
{
    //!@{
    //! \name Type aliases
    using MapIntCutoff = std::map<int, ImportProductionCut>;
    using VecComponent = std::vector<ImportMatElemComponent>;
    //!@}

    std::string name{};
    ImportMaterialState state{ImportMaterialState::size_};
    double temperature;  //!< [K]
    double number_density;  //!< [1/length^3]
    MapIntCutoff pdg_cutoffs;  //!< Cutoff per PDG
    VecComponent elements;
    ImportOpticalMaterial optical_properties;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string label for material state
char const* to_cstring(ImportMaterialState s);

// Get the string label for scalar optical property
char const* to_cstring(ImportOpticalScalar s);

// Get the string label for vector optical property
char const* to_cstring(ImportOpticalVector s);

//---------------------------------------------------------------------------//
}  // namespace celeritas
