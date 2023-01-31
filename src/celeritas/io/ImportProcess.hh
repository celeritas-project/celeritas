//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <string>
#include <vector>

// IWYU pragma: begin_exports
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/io/ImportPhysicsVector.hh"
// IWYU pragma: end_exports

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
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available physics processes.
 *
 * This enum was created to safely access the many physics tables imported.
 */
enum class ImportProcessClass
{
    // User-defined
    unknown,
    // EM
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
    size_
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
    size_
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
 * \c micro_xs.find(model).at(material_id).at(element_id) .
 *
 * Microscopic cross-section data stored in the element-selector physics vector
 * is in cm^2.
 */
struct ImportProcess
{
    //!@{
    //! \name Type aliases
    // One ImportPhysicsVector per element component
    using ElementPhysicsVectors = std::vector<ImportPhysicsVector>;
    // Vector spans over all materials for a given model
    using ModelMicroXS = std::vector<ElementPhysicsVectors>;
    //!@}

    int particle_pdg{0};
    int secondary_pdg{0};
    ImportProcessType process_type;
    ImportProcessClass process_class;
    std::vector<ImportModelClass> models;
    // TODO: map from ImportTableType
    std::vector<ImportPhysicsTable> tables;
    std::map<ImportModelClass, ModelMicroXS> micro_xs;

    explicit operator bool() const
    {
        return process_type != ImportProcessType::not_defined
               && process_class != ImportProcessClass::unknown
               && !models.empty();
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string form of a process enumeration.
char const* to_cstring(ImportProcessType value);
char const* to_cstring(ImportProcessClass value);
char const* to_cstring(ImportModelClass value);

// Get the default Geant4 process name
char const* to_geant_name(ImportProcessClass value);
// Convert a Geant4 process name to an IPC (throw RuntimeError if unsupported)
ImportProcessClass geant_name_to_import_process_class(std::string const& s);

// Whether Celeritas requires microscopic xs data for sampling
bool needs_micro_xs(ImportModelClass model);

//---------------------------------------------------------------------------//
}  // namespace celeritas
