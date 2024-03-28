//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ImportProcess.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
enum class ImportOpticalProcessClass
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
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    ImportOpticalProcess ...;
   \endcode
 */
struct ImportOpticalProcess
{
    const ImportProcessType process_type{ImportProcessType::optical};
    ImportOpticalProcessClass process_class{ImportOpticalProcessClass::size_};
    ImportPhysicsTable lambda_table;

    explicit operator bool() const
    {
        return process_type == ImportProcessType::optical
               && process_class != ImportOpticalProcessClass::size_
               && lambda_table.table_type == ImportTableType::lambda
               && lambda_table;
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string form of one of the enumerations
char const* to_cstring(ImportOpticalProcessClass value);

// Get the default Geant4 process name
char const* to_geant_name(ImportOpticalProcessClass value);
// Convert a Geant4 process name to an IPC (throw RuntimeError if unsupported)
ImportOpticalProcessClass
geant_name_to_import_optical_process_class(std::string_view sv);

//---------------------------------------------------------------------------//
}  // namespace celeritas
