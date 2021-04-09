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
 * \c RootImporter , along with all \c Import[Class] type of classes, are the
 * link between Geant4 and Celeritas. Every Celeritas' host/device class that
 * relies on imported data has its own \c from_import(...) function that will
 * take the data loaded by the \c RootImporter and load it accordingly:
 *
 * \code
 *  RootImporter import("/path/to/root_file.root");
 *  const auto data            = import("tree", "import_data_branch");
 *  const auto particle_params = ParticleParams::from_import(data);
 *  const auto material_params = MaterialParams::from_import(data);
 *  const auto cutoff_params   = CutoffParams::from_import(data);
 *  // And so on
 * \endcode
 */
class RootImporter
{
  public:
    // Construct with ROOT file name
    explicit RootImporter(const char* filename);

    // Release ROOT file on exit
    ~RootImporter();

    // Load data from the ROOT file
    ImportData operator()(const char* tree_name, const char* branch_name);

  private:
    std::unique_ptr<TFile> root_input_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
