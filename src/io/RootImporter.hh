//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/CutoffParams.hh"
#include "physics/material/MaterialParams.hh"

// ROOT
class TFile;

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * RootImporter loads particle, element, material, process, and volume
 * information from a ROOT file that contains an \c ImportData object.
 * Currently, said ROOT file is created by the \e app/geant-exporter external
 * code.
 *
 * The geant-exporter app only pulls data from Geant4. Conversely, all the
 * \c [Class]Params only live in Celeritas. Thus, \c RootImporter , along with
 * all \c Import[Class] type of classes, are the link between Geant4 and
 * Celeritas. Every host/device class that relies on imported data has its own
 * \c from_import(...) function that will take the data loaded by the
 * \c RootImporter and import it accordingly:
 *
 * \code
 *  RootImporter import("/path/to/root_file.root");
 *  const auto data            = import("tree_name", "import_data_branch");
 *  const auto particle_params = ParticleParams::from_import(data);
 *  const auto material_params = MaterialParams::from_import(data);
 *  const auto cutoff_params   = CutoffParams::from_import(data);
 *  // And so on
 * \endcode
 *
 * Material and volume information are stored in the \c GdmlGeometryMap class.
 * The \c mat_id returned from a given \c vol_id represents the position of
 * said material in the \c ImportPhysicsTable vectors:
 * \c ImportPhysicsTable.physics_vectors.at(mat_id) .
 */
class RootImporter
{
  public:
    // Construct with exported ROOT file
    explicit RootImporter(const char* filename);

    // Release ROOT file on exit
    ~RootImporter();

    // Load data from the ROOT file
    ImportData operator()(const char* tree_name, const char* branch_name);

  private:
    //// DATA ////
    std::unique_ptr<TFile> root_input_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
