//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "physics/base/ParticleMd.hh"
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
    ucn
};

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics processes.
 *
 * This enum was created to safely access the many physics tables imported.
 */
enum class ImportProcess
{
    not_defined,
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
    transportation //!< Not a physics process
};

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics models.
 *
 * This enum was created to safely access the many imported physics tables.
 */
enum class ImportModel
{
    not_defined,
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
    mu_pair_prod
};

//---------------------------------------------------------------------------//
/*!
 * Property being described by the physics table.
 *
 * In Geant4 this is a string value.
 */
enum class ImportTableType
{
    not_defined,
    dedx,
    ionisation,
    range,
    range_sec,
    inverse_range,
    lambda,
    lambda_prim,
    lambda_mod_1,
    lambda_mod_2,
    lambda_mod_3,
    lambda_mod_4
};

//---------------------------------------------------------------------------//
/*!
 * Store physics tables.
 *
 * The geant-exporter app stores Geant4 physics tables into a ROOT file, while
 * the RootImporter class is responsible for loading said data into memory.
 */
struct ImportPhysicsTable
{
    ImportProcessType                process_type;
    ImportTableType                  table_type;
    ImportProcess                    process;
    ImportModel                      model;
    PDGNumber                        particle;
    std::vector<ImportPhysicsVector> physics_vectors;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

const char* to_cstring(ImportProcessType value);
const char* to_cstring(ImportProcess value);
const char* to_cstring(ImportModel value);
const char* to_cstring(ImportTableType value);

//---------------------------------------------------------------------------//
} // namespace celeritas
