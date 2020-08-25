//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "GeantParticle.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleDef.hh"
#include "base/Types.hh"
#include "base/Macros.hh"

// ROOT forward declarations
class TFile;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * GeantImporter loads particle and physics table data from the ROOT file
 * created by the app/geant-exporter external code.
 *
 * Usage:
 * 1. The constructor takes the /path/to/rootfile.root as a parameter and
 *    opens the root file
 * 2. Operator() loads particle and table data into a result_type struct
 *    that contains shared pointers to ParticleParams and GeantPhysicsTables
 *
 * \code
 *  GeantImporter import("/path/to/rootfile.root");
 *  auto geant_data = import();
 * \endcode
 */
class GeantImporter
{
  public:
    struct result_type
    {
        std::shared_ptr<ParticleParams> particle_params;
    };

  public:
    // Construct with exported ROOT file
    explicit GeantImporter(const char* filename);

    // Release ROOT file on exit
    ~GeantImporter();

    // Load data from the ROOT file into result_type
    result_type operator()();

  private:
    // Populate the shared_ptr<ParticleParams> with particle information
    std::shared_ptr<ParticleParams> load_particle_data();

  public:
    std::unique_ptr<TFile> root_input_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
