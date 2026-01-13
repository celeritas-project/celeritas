//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Particle production cutoff values: range and approximate energy.
 */
struct ImportProductionCut
{
    double energy{};  //!< [MeV]
    double range{};  //!< [length]
};

//---------------------------------------------------------------------------//
/*!
 * Fractional elemental composition of a given material.
 */
struct ImportMatElemComponent
{
    //!@{
    //! \name Type aliases
    using ElIndex = unsigned int;
    //!@}

    ElIndex element_id{};  //!< Index of element in ImportElement
    double number_fraction{};  //!< [unitless]
};

//---------------------------------------------------------------------------//
/*!
 * Material data as specified by a geometry model.
 *
 * These are the "real life properties" unaffected by changes to the user's
 * physics selection.
 */
struct ImportGeoMaterial
{
    //!@{
    //! \name Type aliases
    using VecComponent = std::vector<ImportMatElemComponent>;
    //!@}

    std::string name{};
    ImportMaterialState state{ImportMaterialState::other};
    double temperature{};  //!< [K]
    double number_density{};  //!< [1/length^3]
    VecComponent elements;
};

//---------------------------------------------------------------------------//
/*!
 * Distinct materials as modified by physics.
 *
 * User-selected regions can alter physics properties so that the same
 * "geometry material" can correspond to multiple "physics materials". These
 * include the behavior of the material as an optical region.
 *
 * - \c geo_material_id is a geometry material corresponding to the index in
 *   the \c ImportData.geo_materials
 * - \c optical_material_id is an \em optional optical material corresponding
 *   to the index in the \c ImportData.optical_materials
 *
 * Geant4 requires an optical material to correspond to a single geo material,
 * but we may relax this restriction in the future.
 */
struct ImportPhysMaterial
{
    //!@{
    //! \name Type aliases
    using Index = unsigned int;
    using PdgInt = int;
    using MapIntCutoff = std::map<PdgInt, ImportProductionCut>;
    //!@}

    static constexpr Index unspecified = static_cast<Index>(-1);

    Index geo_material_id{};  //!< Index in geo_materials list
    Index optical_material_id{unspecified};  //!< Optional index in optical mat
    MapIntCutoff pdg_cutoffs;  //!< Cutoff per PDG
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string label for material state
char const* to_cstring(ImportMaterialState s);

//---------------------------------------------------------------------------//
}  // namespace celeritas
