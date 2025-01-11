//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/GeantImport.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

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
 */
struct GeantImport
{
    //! Do not use Celeritas physics for the given Geant4 process names
    std::vector<std::string> ignore_processes;

    // TODO: other GeantImportDataSelection options for em/optical?
};

//---------------------------------------------------------------------------//
/*!
 * Options for loading cross section data from Geant4 data files.
 *
 * \todo This is not yet used, but it will call LivermorePEReader,
 * SeltzerBergerReader, AtomicRelaxationReader to fill cross section data.
 * Since Geant4 data structures don't provide access to these, we must read
 * them ourselves.
 */
struct GeantDataImport
{
    //! Livermore photoelectric data directory (G4LEDATA)
    std::string livermore_dir;
    //! Particle cross section data directory (G4PARTICLEXS)
    std::string particle_dir;
};

//---------------------------------------------------------------------------//
/*!
 * Update tuning and diagnostic options from an external input file.
 *
 * This is used in concert with \c FileImport : the output from another code
 * can be used as input, but overlaid with diagnostic and tuning information.
 */
struct UpdateImport
{
    bool diagnostics{true};
    bool tuning{true};

    //! Path to the file
    std::string input;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
