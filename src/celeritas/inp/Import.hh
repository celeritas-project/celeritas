//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Import.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "celeritas/ext/GeantImporter.hh"

#include "System.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
struct Problem;
struct System;

//---------------------------------------------------------------------------//
/*!
 * Options for loading problem data from a ROOT/JSON file.
 */
struct FileImport
{
    //! Path to the problem input file
    std::string input;
};

//---------------------------------------------------------------------------//
/*!
 * Options for importing data from in-memory Geant4.
 *
 * \todo Update \c GeantImportDataSelection to control what Celeritas
 * processes/particles are offloaded.
 */
struct GeantImport
{
    //! Do not use Celeritas physics for the given Geant4 process names
    std::vector<std::string> ignore_processes;

    //! Only import a subset of available Geant4 data
    GeantImportDataSelection data_selection;
};

//---------------------------------------------------------------------------//
/*!
 * Options for loading cross section data from Geant4 data files.
 *
 * \todo This is not yet used, but it will call LivermorePEReader,
 * SeltzerBergerReader, AtomicRelaxationReader to fill cross section data.
 * Since Geant4 data structures don't provide access to these, we must read
 * them ourselves.
 *
 * Defaults:
 * - \c livermore_dir: uses the \c G4LEDATA environment variable
 * - \c particle_dir: uses the \c G4PARTICLEXS environment variable
 */
struct GeantDataImport
{
    //! Livermore photoelectric data directory (optional)
    std::string livermore_dir;
    //! Particle cross section data directory (optional)
    std::string particle_dir;
};

//---------------------------------------------------------------------------//
/*!
 * Update control and diagnostic options from an external input file.
 *
 * This is used in concert with \c FileImport : the output from another code
 * can be used as input, but overlaid with diagnostic and control/tuning
 * information.
 */
struct UpdateImport
{
    //! Replace existing diagnostics
    bool diagnostics{true};
    //! Replace existing control parameters
    bool control{true};

    //! Path to the file
    std::string input;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
