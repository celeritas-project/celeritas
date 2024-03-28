//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportOpticalProcess.cc
//---------------------------------------------------------------------------//
#include "ImportOpticalProcess.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/StringEnumMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enumerator for the available optical physics processes.
 *
 * This enum was created to safely access the many physics tables imported.
 */
char const* to_cstring(ImportOpticalProcessClass value)
{
    static EnumStringMapper<ImportOpticalProcessClass> const to_cstring_impl{
        "",
        "absorption",
        "rayleigh",
        "wls",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get the default Geant4 process name for an ImportOpticalProcessClass.
 */
char const* to_geant_name(ImportOpticalProcessClass value)
{
    static EnumStringMapper<ImportOpticalProcessClass> const to_name_impl{
        "",  // unknown,
        "absorption",
        "rayleigh",
        "wls",
    };
    return to_name_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 process name to an IPC.
 *
 * This will throw a \c celeritas::RuntimeError if the string is not known to
 * us.
 */
ImportOpticalProcessClass
geant_name_to_import_optical_process_class(std::string_view s)
{
    static auto const from_string
        = StringEnumMapper<ImportOpticalProcessClass>::from_cstring_func(
            to_geant_name, "optical process class");

    return from_string(s);
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
