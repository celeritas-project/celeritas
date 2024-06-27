//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalModel.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enum of types of optical models that may be imported.
 */
enum class ImportOpticalModelClass
{
    other,
    // Optical
    absorption,
    rayleigh,
    wavelength_shifting,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Store imported physics data associated with a given optical model.
 */
struct ImportOpticalModel
{
    ImportProcessType const process_type{ImportProcessType::optical};
    ImportOpticalModelClass model_class{ImportOpticalModelClass::size_};
    ImportPhysicsTable lambda_table;

    explicit operator bool() const
    {
        return process_type == ImportProcessType::optical
            && model_class != ImportOpticalModelClass::size_
            && lambda_table.table_type == ImportTableType::lambda
            && lambda_table;
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string form of the enumeration
char const* to_cstring(ImportOpticalModelClass value);

// Get the default Geant4 name for the optical model
char const* to_geant_name(ImportOpticalModelClass value);
// Convert a Geant4 process name to an IPC (throw RuntimeError if unsupported)
ImportOpticalModelClass geant_name_to_import_optical_model_class(std::string_view sv);

//---------------------------------------------------------------------------//
}  // namespace celeritas
