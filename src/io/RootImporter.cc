//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootImporter.cc
//---------------------------------------------------------------------------//
#include "RootImporter.hh"

#include <iomanip>
#include <iostream>
#include <tuple>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>

#include "base/Assert.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Constructor
 */
RootImporter::RootImporter(const char* filename)
{
    root_input_.reset(TFile::Open(filename, "read"));
    ENSURE(root_input_);
}

//---------------------------------------------------------------------------//
//! Default destructor
RootImporter::~RootImporter() = default;

//---------------------------------------------------------------------------//
/*!
 * Load all data from the input file
 */
RootImporter::result_type RootImporter::operator()()
{
    result_type geant_data;
    geant_data.particle_params = this->load_particle_data();
    geant_data.physics_tables  = this->load_physics_table_data();
    geant_data.geometry        = this->load_geometry_data();

    ENSURE(geant_data.particle_params);
    ENSURE(geant_data.physics_tables);
    ENSURE(geant_data.geometry);

    return geant_data;
}

//---------------------------------------------------------------------------//
/*!
 * Load all ImportParticle objects from the ROOT file as ParticleParams
 */
std::shared_ptr<ParticleParams> RootImporter::load_particle_data()
{
    // Open the 'particles' branch and reserve size for the converted data
    std::unique_ptr<TTree> tree_particles(
        root_input_->Get<TTree>("particles"));
    CHECK(tree_particles);

    ParticleParams::VecAnnotatedDefs defs(tree_particles->GetEntries());
    CHECK(!defs.empty());

    // Load the particle data
    ImportParticle  particle;
    ImportParticle* temp_particle_ptr = &particle;
    tree_particles->SetBranchAddress("ImportParticle", &temp_particle_ptr);
    for (auto i : range(defs.size()))
    {
        // Load a single entry into particle
        particle.name.clear();
        tree_particles->GetEntry(i);
        CHECK(!particle.name.empty());

        // Convert metadata
        ParticleMd particle_md;
        particle_md.name     = particle.name;
        particle_md.pdg_code = PDGNumber{particle.pdg};
        CHECK(particle_md.pdg_code);
        defs[i].first = std::move(particle_md);

        // Convert data
        ParticleDef particle_def;
        particle_def.mass   = units::MevMass{particle.mass};
        particle_def.charge = units::ElementaryCharge{particle.charge};
        particle_def.decay_constant
            = (particle.is_stable ? ParticleDef::stable_decay_constant()
                                  : 1. / particle.lifetime);
        defs[i].second = std::move(particle_def);
    }

    // Sort by increasing mass, then by PDG code. Placing lighter particles
    // (more likely to be created by various processes, so more "light
    // particle" tracks) together at the beginning of the list will make it
    // easier to human-read the particles while debugging, and having them
    // at adjacent memory locations could improve cacheing.
    std::sort(defs.begin(), defs.end(), [](const auto& lhs, const auto& rhs) {
        return std::make_tuple(lhs.second.mass, lhs.first.pdg_code)
               < std::make_tuple(rhs.second.mass, rhs.first.pdg_code);
    });

    // Construct ParticleParams from the definitions
    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//
/*!
 * Load all ImportPhysicsTable objects from the ROOT file as a vector
 */
std::shared_ptr<std::vector<ImportPhysicsTable>>
RootImporter::load_physics_table_data()
{
    // Open tables branch
    std::unique_ptr<TTree> tree_tables(root_input_->Get<TTree>("tables"));
    CHECK(tree_tables);
    CHECK(tree_tables->GetEntries());

    // Load branch
    ImportPhysicsTable  a_table;
    ImportPhysicsTable* temp_table_ptr = &a_table;
    tree_tables->SetBranchAddress("ImportPhysicsTable", &temp_table_ptr);

    std::vector<ImportPhysicsTable> tables;

    // Populate physics table vector
    for (size_type i : range(tree_tables->GetEntries()))
    {
        tree_tables->GetEntry(i);
        tables.push_back(a_table);
    }

    return std::make_shared<std::vector<ImportPhysicsTable>>(std::move(tables));
}
//---------------------------------------------------------------------------//
/*!
 * Load GdmlGeometryMap from the ROOT file
 */
std::shared_ptr<GdmlGeometryMap> RootImporter::load_geometry_data()
{
    // Open geometry branch
    std::unique_ptr<TTree> tree_geometry(root_input_->Get<TTree>("geometry"));
    CHECK(tree_geometry);
    CHECK(tree_geometry->GetEntries()); // Must be 1

    // Load branch and fetch data
    GdmlGeometryMap  geometry;
    GdmlGeometryMap* geometry_ptr = &geometry;
    tree_geometry->SetBranchAddress("GdmlGeometryMap", &geometry_ptr);
    tree_geometry->GetEntry(0);

    // The info from the GdmlGeometryMap will be used by a class that will be
    // constructed on the host and will manage host/device data

    return std::make_shared<GdmlGeometryMap>(std::move(geometry));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
