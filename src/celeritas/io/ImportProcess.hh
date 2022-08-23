//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

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
    bethe_heitler,
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
 * \c ImportPhysicsTable is process and type (lambda, dedx, and so
 * on) dependent, with each table type including physics vectors for all
 * materials. Therefore, the physics vector of a given material is retrieved
 * by finding the appropriate \c table_type in the \c tables vector and
 * selecting the material: \c table.physics_vectors.at(material_id) .
 *
 * Conversely, element-selectors are model dependent. Thus, for simplicity,
 * they are stored directly as physics vectors and retrieved by providing the
 * model class enum, material, and element id:
 * \c micro_xs.find(model).at(material_id).find(element_id)->second .
 *
 * Microscopic cross-section data stored in the element-selector physics vector
 * is in cm^2.
 */
struct ImportProcess
{
    //!@{
    //! Type aliases
    // One map per material: <element_id, physics_vector>
    using ElementPhysicsVectorMap = std::map<int, ImportPhysicsVector>;
    // Vector spans over all materials for a given model
    using ModelMicroXS = std::vector<ElementPhysicsVectorMap>;
    //!@}

    int                                      particle_pdg{0};
    int                                      secondary_pdg{0};
    ImportProcessType                        process_type;
    ImportProcessClass                       process_class;
    std::vector<ImportModelClass>            models;
    std::vector<ImportPhysicsTable>          tables;
    std::map<ImportModelClass, ModelMicroXS> micro_xs;

    explicit operator bool() const
    {
        return process_type != ImportProcessType::not_defined
               && process_class != ImportProcessClass::unknown
               && !models.empty() && !tables.empty();
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
