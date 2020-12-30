//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "physics/base/PDGNumber.hh"
#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Category of physics process.
 *
 * See Geant4's G4ProcessType.hh for the equivalent enum.
 */
enum class ImportProcessType
{
    not_defined,
    transportation,
    electromagnetic,
    optical,
    hadronic,
    photolepton_hadron,
    decay,
    general,
    parameterisation,
    user_defined,
    parallel,
    phonon,
    ucn,
};

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics processes.
 *
 * This enum was created to safely access the many physics tables imported.
 */
enum class ImportProcess
{
    unknown,
    ion_ioni,
    msc,
    h_ioni,
    h_brems,
    h_pair_prod,
    coulomb_scat,
    e_ioni,
    e_brem,
    photoelectric,
    compton,
    conversion,
    rayleigh,
    annihilation,
    mu_ioni,
    mu_brems,
    mu_pair_prod,
};

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics models.
 *
 * This enum was created to safely access the many imported physics tables.
 */
enum class ImportModel
{
    unknown,
    bragg_ion,
    bethe_bloch,
    urban_msc,
    icru_73_qo,
    wentzel_VI_uni,
    h_brem,
    h_pair_prod,
    e_coulomb_scattering,
    bragg,
    moller_bhabha,
    e_brem_sb,
    e_brem_lpm,
    e_plus_to_gg,
    livermore_photoelectric,
    klein_nishina,
    bethe_heitler_lpm,
    livermore_rayleigh,
    mu_bethe_bloch,
    mu_brem,
    mu_pair_prod,
};

//---------------------------------------------------------------------------//
/*!
 * Property being described by the physics table.
 *
 * These are named based on accessors in
 */
enum class ImportTableType
{
    dedx,
    dedx_subsec,
    dedx_unrestricted,
    ionisation,
    ionisation_subsec,
    csda_range, //!< Continuous slowing down approximation
    range,
    secondary_range,
    inverse_range,
    lambda,      //!< Macroscopic cross section
    sublambda,   //!< For subcutoff regions
    lambda_prim, //!< Cross section scaled by energy
};

//---------------------------------------------------------------------------//
/*!
 * Units of a physics table.
 */
enum class ImportUnits
{
    none,       //!< Unitless
    cm_inv,     //!< Macroscopic xs (1/cm)
    cm_mev_inv, //!< Macroscopic xs divided by energy (1/cm-MeV)
    mev,        //!< Energy loss (MeV)
    cm,         //!< Range (cm)
};

//---------------------------------------------------------------------------//
/*!
 * Imported physics table.
 *
 * The geant-exporter app stores Geant4 physics tables into a ROOT file, while
 * the RootImporter class is responsible for loading said data into memory.
 */
struct ImportPhysicsTable
{
    PDGNumber                        particle;
    ImportProcessType                process_type;
    ImportProcess                    process;
    ImportModel                      model;
    ImportTableType                  table_type;
    ImportUnits                      units;
    std::vector<ImportPhysicsVector> physics_vectors;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

const char* to_cstring(ImportProcessType value);
const char* to_cstring(ImportProcess value);
const char* to_cstring(ImportModel value);
const char* to_cstring(ImportTableType value);
const char* to_cstring(ImportUnits value);

//---------------------------------------------------------------------------//
} // namespace celeritas
