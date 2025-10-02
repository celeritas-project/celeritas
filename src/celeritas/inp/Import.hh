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
 * Load physics data from a ROOT file.
 *
 * \todo This should be replaced with a "ProblemFromFile" that supports ROOT or
 * JSON. Currently it loads directly into \c ImportData as a stopgap. We may
 * also want to completely replace ROOT.
 */
struct PhysicsFromFile
{
    //! Path to the problem input file
    std::string input;
};

//---------------------------------------------------------------------------//
/*!
 * Options for importing data from in-memory Geant4.
 *
 * \todo
 * - Use "offload particle types" (variant: grouping, G4PD*, PDG)
 * - Load all processes applicable to offload particles
 * - Determine particle list from process->secondary mapping
 * - Always load interpolation flags; clear them elsewhere if user wants to
 * - Load all materials visible to geometry (and eventually fix PhysMatId vs
 *   GeoMatId)
 */
struct PhysicsFromGeant
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
 * \todo Since Geant4 data structures don't provide access to these, we must
 * read them ourselves. Maybe add accessors to Geant4 and eliminate these/roll
 * them upstream?
 *
 * Defaults:
 * - \c livermore_dir: usually <code>$G4LEDATA/livermore/phot_epics2014</code>
 * - \c neutron_dir: usually <code>$G4PARTICLEXSDATA/neutron</code>
 * - \c fluor_dir: usually <code>$G4LEDATA/fluor</code>
 * - \c auger_dir: usually <code>$G4LEDATA/auger</code>
 */
struct PhysicsFromGeantFiles
{
    //! Livermore photoelectric data directory
    std::string livermore_dir;
    //! Neutron cross section data directory
    std::string neutron_dir;
    //! Fluorescence transition probabilities and subshells
    std::string fluor_dir;
    //! Auger transition probabilities
    std::string auger_dir;
};

//---------------------------------------------------------------------------//
/*!
 * \todo Add a class to update control and diagnostic options from an external
 * input file.
 *
 * This will be used in concert with \c FileImport : the output from another
 * code can be used as input, but overlaid with diagnostic and control/tuning
 * information.
 */

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
