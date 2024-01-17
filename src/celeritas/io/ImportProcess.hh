//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <string_view>
#include <vector>

#include "corecel/Macros.hh"
// IWYU pragma: begin_exports
#include "celeritas/io/ImportModel.hh"
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
    other,
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
    other,
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
    gamma_general,  // Will be decomposed into other processes
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Store physics process data.
 *
 * \note
 * \c ImportPhysicsTable is process and type (lambda, dedx, and so
 * on) dependent, with each table type including physics vectors for all
 * materials. Therefore, the physics vector of a given material is retrieved
 * by finding the appropriate \c table_type in the \c tables vector and
 * selecting the material: \c table.physics_vectors.at(material_id) .
 */
struct ImportProcess
{
    int particle_pdg{0};
    int secondary_pdg{0};
    ImportProcessType process_type{ImportProcessType::size_};
    ImportProcessClass process_class{ImportProcessClass::size_};
    std::vector<ImportModel> models;
    std::vector<ImportPhysicsTable> tables;

    explicit operator bool() const
    {
        return particle_pdg != 0 && process_type != ImportProcessType::size_
               && process_class != ImportProcessClass::size_ && !models.empty();
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string form of one of the enumerations
char const* to_cstring(ImportProcessType value);
char const* to_cstring(ImportProcessClass value);

// Get the default Geant4 process name
char const* to_geant_name(ImportProcessClass value);
// Convert a Geant4 process name to an IPC (throw RuntimeError if unsupported)
ImportProcessClass geant_name_to_import_process_class(std::string_view sv);

//---------------------------------------------------------------------------//
}  // namespace celeritas
