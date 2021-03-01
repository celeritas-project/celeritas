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
#include "physics/material/MaterialParams.hh"
#include "ImportProcess.hh"
#include "GdmlGeometryMap.hh"

// ROOT
class TFile;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * RootImporter loads particle, physics table, material, and geometry
 * data from the ROOT file created by the app/geant-exporter external code.
 *
 * The geant-exporter app only pulls data from Geant4, and we will keep it
 * that way for validation and comparison purposes. Conversely, MaterialParams
 * is a Celeritas class. Thus, RootImporter acts as a bridge and is responsible
 * for converting any quantity between Geant4 -> Celeritas.
 *
 * Usage:
 * \code
 *  RootImporter import("/path/to/rootfile.root");
 *  auto geant_data = import();
 * \endcode
 *
 * Physics tables currently are a vector<ImportPhysicsTable>, since many
 * parameters are at play when selecting a given table:
 * ImportParticle, ImportTableType, ImportProcessClass, and ImportModelClass.
 * See RootImporter.test.cc for an example on how to fetch a given table.
 * This method will probably have to be improved.
 *
 * Material and volume information are stored in a GdmlGeometryMap object.
 * The GdmlGeometryMap::mat_id value returned from a given vol_id represents
 * the position of said material in the ImportPhysicsTable vectors:
 * \c ImportPhysicsTable.physics_vectors.at(mat_id_value).
 */
class RootImporter
{
  public:
    //! Return value from importing from the ROOT file
    struct result_type
    {
        std::shared_ptr<ParticleParams>  particle_params;
        std::vector<ImportProcess>       processes;
        std::shared_ptr<GdmlGeometryMap> geometry;
        std::shared_ptr<MaterialParams>  material_params;
    };

  public:
    // Construct with exported ROOT file
    explicit RootImporter(const char* filename);

    // Release ROOT file on exit
    ~RootImporter();

    // Load data from the ROOT file into result_type
    result_type operator()();

  private:
    //// DATA ////
    std::unique_ptr<TFile> root_input_;

    //// HELPER FUNCTIONS ////

    // Populate the shared_ptr<ParticleParams> with particle information
    std::shared_ptr<ParticleParams> load_particle_data();
    // Populate a vector of ImportPhysicsTable objects
    std::vector<ImportProcess> load_processes();
    // Load GdmlGeometryMap object
    std::shared_ptr<GdmlGeometryMap> load_geometry_data();
    // Populate the shared_ptr<MaterialParams> with material information
    std::shared_ptr<MaterialParams> load_material_data();
    // Safely switch between Import and Celeritas enums
    MaterialParams::ProductionCutId
    to_production_cut_id(const ImportProductionCut& value);
};

//---------------------------------------------------------------------------//
} // namespace celeritas
