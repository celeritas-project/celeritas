//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "physics/base/PDGNumber.hh"
#include "ImportPhysicsTable.hh"

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
enum class ImportProcessClass
{
    unknown,
    ion_ioni,
    msc,
    h_ioni,
    h_brems,
    h_pair_prod,
    coulomb_scat,
    e_ioni,
    e_brems,
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
enum class ImportModelClass
{
    unknown,
    bragg_ion,
    bethe_bloch,
    urban_msc,
    icru_73_qo,
    wentzel_VI_uni,
    h_brems,
    h_pair_prod,
    e_coulomb_scattering,
    bragg,
    moller_bhabha,
    e_brems_sb,
    e_brems_lpm,
    e_plus_to_gg,
    livermore_photoelectric,
    klein_nishina,
    bethe_heitler_lpm,
    livermore_rayleigh,
    mu_bethe_bloch,
    mu_brems,
    mu_pair_prod,
};

//---------------------------------------------------------------------------//
/*!
 * Store physics process data.
 *
 * \sa ImportData
 *
 * \note
 * XS tables depend on the physics process and their type (lambda, dedx, and so
 * on). Each table type includes physics vectors for all materials. Therefore,
 * the XS physics vector of a given material is retrieved by
 * doing \c tables.at(table_type).physics_vectors.at(material_id) .
 * Conversely, element selector physics vectors are model dependent. Thus, for
 * simplicity, they are stored directly as physics vectors and retrieved by
 * calling \c element_selector_tables.at(model_id).at(material_id)
 */
struct ImportProcess
{
    int                                           particle_pdg;
    ImportProcessType                             process_type;
    ImportProcessClass                            process_class;
    std::vector<ImportModelClass>                 models;
    std::vector<ImportPhysicsTable>               tables;
    std::vector<std::vector<ImportPhysicsVector>> element_selector_tables;

    explicit operator bool() const
    {
        return process_type != ImportProcessType::not_defined
               && process_class != ImportProcessClass::unknown
               && !models.empty() && !tables.empty()
               && !element_selector_tables.empty();
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

const char* to_cstring(ImportProcessType value);
const char* to_cstring(ImportProcessClass value);
const char* to_cstring(ImportModelClass value);

//---------------------------------------------------------------------------//
} // namespace celeritas
