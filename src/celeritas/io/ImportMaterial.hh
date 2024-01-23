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
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string label for material state
char const* to_cstring(ImportMaterialState s);

//---------------------------------------------------------------------------//
}  // namespace celeritas
