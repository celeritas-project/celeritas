//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootImporter.cc
//---------------------------------------------------------------------------//
#include "RootImporter.hh"

#include <cstdlib>
#include <iomanip>
#include <tuple>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>

#include "base/Assert.hh"
#include "base/Range.hh"
#include "comm/Logger.hh"
#include "physics/base/Units.hh"
#include "ImportParticle.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Safely switch between MatterState [MaterialParams.hh] and
 * ImportMaterialState [ImportMaterial.hh].
 */
MatterState to_matter_state(const ImportMaterialState state)
{
    switch (state)
    {
        case ImportMaterialState::not_defined:
            return MatterState::unspecified;
        case ImportMaterialState::solid:
            return MatterState::solid;
        case ImportMaterialState::liquid:
            return MatterState::liquid;
        case ImportMaterialState::gas:
            return MatterState::gas;
    }
    CELER_ASSERT_UNREACHABLE();
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from path to ROOT file.
 */
RootImporter::RootImporter(const char* filename)
{
    CELER_LOG(status) << "Opening ROOT file";
    root_input_.reset(TFile::Open(filename, "read"));
    CELER_ENSURE(root_input_ && !root_input_->IsZombie());
}

//---------------------------------------------------------------------------//
//! Default destructor
RootImporter::~RootImporter() = default;

//---------------------------------------------------------------------------//
/*!
 * Load all data from the input file.
 */
RootImporter::result_type RootImporter::operator()()
{
    result_type geant_data;
    geant_data.particle_params = this->load_particle_data();
    geant_data.processes       = this->load_processes();
    geant_data.geometry        = this->load_geometry_data();
    geant_data.material_params = this->load_material_data();
    geant_data.cutoff_params   = this->load_cutoff_data();

    // Sort processes based on particle def IDs, process types, etc.
    {
        const ParticleParams& particles = *geant_data.particle_params;
        auto to_process_key = [&particles](const ImportProcess& ip) {
            return std::make_tuple(particles.find(PDGNumber{ip.particle_pdg}),
                                   ip.process_class);
        };
        std::sort(geant_data.processes.begin(),
                  geant_data.processes.end(),
                  [&to_process_key](const ImportProcess& left,
                                    const ImportProcess& right) {
                      return to_process_key(left) < to_process_key(right);
                  });
    }

    CELER_ENSURE(geant_data.particle_params);
    CELER_ENSURE(geant_data.geometry);
    CELER_ENSURE(geant_data.material_params);
    CELER_ENSURE(geant_data.cutoff_params);
    return geant_data;
}

//---------------------------------------------------------------------------//
/*!
 * Load all ImportParticle objects from the ROOT file as ParticleParams.
 */
std::shared_ptr<ParticleParams> RootImporter::load_particle_data()
{
    CELER_LOG(status) << "Loading particle data";
    // Open the 'particles' branch and reserve size for the converted data
    std::unique_ptr<TTree> tree_particles(
        root_input_->Get<TTree>("particles"));
    CELER_ASSERT(tree_particles);

    ParticleParams::Input defs(tree_particles->GetEntries());
    CELER_ASSERT(!defs.empty());

    // Load the particle data
    ImportParticle  particle;
    ImportParticle* temp_particle_ptr = &particle;

    int err_code = tree_particles->SetBranchAddress("ImportParticle",
                                                    &temp_particle_ptr);
    CELER_ASSERT(err_code >= 0);

    for (auto i : range(defs.size()))
    {
        // Load a single entry into particle
        particle.name.clear();
        tree_particles->GetEntry(i);
        CELER_ASSERT(!particle.name.empty());

        // Convert metadata
        defs[i].name     = particle.name;
        defs[i].pdg_code = PDGNumber{particle.pdg};
        CELER_ASSERT(defs[i].pdg_code);

        // Convert data
        defs[i].mass           = units::MevMass{particle.mass};
        defs[i].charge         = units::ElementaryCharge{particle.charge};
        defs[i].decay_constant = (particle.is_stable
                                      ? ParticleDef::stable_decay_constant()
                                      : 1. / particle.lifetime);
    }

    // Sort by increasing mass, then by PDG code (positive before negative of
    // the same absolute value). Placing lighter particles
    // (more likely to be created by various processes, so more "light
    // particle" tracks) together at the beginning of the list will make it
    // easier to human-read the particles while debugging, and having them
    // at adjacent memory locations could improve cacheing.
    auto to_particle_key = [](const auto& inp) {
        int pdg = inp.pdg_code.get();
        return std::make_tuple(inp.mass, std::abs(pdg), pdg < 0);
    };
    std::sort(defs.begin(),
              defs.end(),
              [to_particle_key](const auto& lhs, const auto& rhs) {
                  return to_particle_key(lhs) < to_particle_key(rhs);
              });

    // Construct ParticleParams from the definitions
    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//
/*!
 * Load all ImportProcess objects from the ROOT file as a vector.
 */
std::vector<ImportProcess> RootImporter::load_processes()
{
    CELER_LOG(status) << "Loading physics processes";
    std::unique_ptr<TTree> tree_processes(
        root_input_->Get<TTree>("processes"));
    CELER_ASSERT(tree_processes);
    CELER_ASSERT(tree_processes->GetEntries());

    // Load branch
    ImportProcess  process;
    ImportProcess* process_ptr = &process;

    int err_code
        = tree_processes->SetBranchAddress("ImportProcess", &process_ptr);
    CELER_ASSERT(err_code >= 0);

    std::vector<ImportProcess> processes;

    // Populate physics process vector
    for (size_type i : range(tree_processes->GetEntries()))
    {
        tree_processes->GetEntry(i);
        processes.push_back(process);
    }

    return processes;
}
//---------------------------------------------------------------------------//
/*!
 * [TEMPORARY] Load GdmlGeometryMap object from the ROOT file.
 *
 * For fully testing the loaded geometry information only.
 *
 * It will be removed as soon as we can load both MATERIAL and VOLUME
 * information into host/device classes.
 */
std::shared_ptr<GdmlGeometryMap> RootImporter::load_geometry_data()
{
    CELER_LOG(status) << "Loading geometry data";
    // Open geometry branch
    std::unique_ptr<TTree> tree_geometry(root_input_->Get<TTree>("geometry"));
    CELER_ASSERT(tree_geometry);
    CELER_ASSERT(tree_geometry->GetEntries()); // Must be 1

    // Load branch and fetch data
    GdmlGeometryMap  geometry;
    GdmlGeometryMap* geometry_ptr = &geometry;

    int err_code
        = tree_geometry->SetBranchAddress("GdmlGeometryMap", &geometry_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_geometry->GetEntry(0);

    return std::make_shared<GdmlGeometryMap>(std::move(geometry));
}

//---------------------------------------------------------------------------//
/*!
 * Load GdmlGeometryMap from the ROOT file and populate MaterialParams.
 */
std::shared_ptr<MaterialParams> RootImporter::load_material_data()
{
    CELER_LOG(status) << "Loading material data";
    // Open geometry branch
    std::unique_ptr<TTree> tree_geometry(root_input_->Get<TTree>("geometry"));
    CELER_ASSERT(tree_geometry);
    CELER_ASSERT(tree_geometry->GetEntries() == 1);

    // Load branch and fetch data
    GdmlGeometryMap  geometry;
    GdmlGeometryMap* geometry_ptr = &geometry;

    int err_code
        = tree_geometry->SetBranchAddress("GdmlGeometryMap", &geometry_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_geometry->GetEntry(0);

    // Create MaterialParams input for its constructor
    MaterialParams::Input input;

    // Populate input.elements
    for (const auto& elem_key : geometry.elemid_to_element_map())
    {
        MaterialParams::ElementInput element_params;
        element_params.atomic_number = elem_key.second.atomic_number;
        element_params.atomic_mass
            = units::AmuMass{elem_key.second.atomic_mass};
        element_params.name = elem_key.second.name;

        input.elements.push_back(element_params);
    }

    // Populate input.materials
    for (const auto& mat_key : geometry.matid_to_material_map())
    {
        MaterialParams::MaterialInput material_params;
        material_params.name           = mat_key.second.name;
        material_params.temperature    = mat_key.second.temperature;
        material_params.number_density = mat_key.second.number_density;
        material_params.matter_state   = to_matter_state(mat_key.second.state);

        for (const auto& elem_key : mat_key.second.elements_num_fractions)
        {
            ElementId elem_def_id{elem_key.first};

            // Populate MaterialParams number fractions
            material_params.elements_fractions.push_back(
                {elem_def_id, elem_key.second});
        }
        /*
        for (const auto& cut_key : mat_key.second.range_cuts)
        {
            const auto prod_cut_id = this->to_production_cut_id(cut_key.first);

            MaterialParams::ProductionCuts prod_cuts;
            prod_cuts.range  = cut_key.second;
            prod_cuts.energy = units::MevEnergy{
                mat_key.second.energy_cuts.find(cut_key.first)->second};

            material_params.production_cuts.insert({prod_cut_id, prod_cuts});
        }
        */
        input.materials.push_back(material_params);
    }

    // Construct MaterialParams and return it as a shared_ptr
    MaterialParams materials(input);
    return std::make_shared<MaterialParams>(std::move(materials));
}

//---------------------------------------------------------------------------//
/*!
 * Load Cutoff information from the ROOT file and populate CutoffParams.
 */
std::shared_ptr<CutoffParams> RootImporter::load_cutoff_data()
{
    CELER_LOG(status) << "Loading cutoff data";

    // Load geometry tree
    std::unique_ptr<TTree> tree_geometry(root_input_->Get<TTree>("geometry"));
    CELER_ASSERT(tree_geometry);
    CELER_ASSERT(tree_geometry->GetEntries() == 1);

    // Load branch and fetch data
    GdmlGeometryMap  geometry;
    GdmlGeometryMap* geometry_ptr = &geometry;

    int err_code
        = tree_geometry->SetBranchAddress("GdmlGeometryMap", &geometry_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_geometry->GetEntry(0);

    // Load particle tree
    std::unique_ptr<TTree> tree_particles(
        root_input_->Get<TTree>("particles"));
    CELER_ASSERT(tree_particles);

    // Load the particle data
    ImportParticle  particle;
    ImportParticle* temp_particle_ptr = &particle;

    err_code = tree_particles->SetBranchAddress("ImportParticle",
                                                &temp_particle_ptr);
    CELER_ASSERT(err_code >= 0);

    CutoffParams::Input input;

    for (const auto& mat_key : geometry.matid_to_material_map())
    {
        // Acess cutoff data from root file
        auto import_material = mat_key.second;

        CutoffParams::MaterialCutoff material_cutoff;

        for (int entry = 0; entry < tree_particles->GetEntries(); ++entry)
        {
            tree_particles->GetEntry(entry);
            SingleCutoff particle_cutoff = {units::MevEnergy{0}, 0};
            auto         iter = import_material.pdg_cutoff.find(particle.pdg);

            if (iter != import_material.pdg_cutoff.end())
            {
                particle_cutoff = {units::MevEnergy{iter->second.energy},
                                   iter->second.range};
            }
            material_cutoff.push_back(particle_cutoff);
        }
        input.push_back(material_cutoff);
    }

    CutoffParams cutoffs(input);
    return std::make_shared<CutoffParams>(std::move(cutoffs));
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch between ImportProductionCut to ProductionCutId enums.
 */
/*
MaterialParams::ProductionCutId
RootImporter::to_production_cut_id(const ImportProductionCut& value)
{
    switch (value)
    {
        case ImportProductionCut::gamma:
            return MaterialParams::ProductionCutId::gamma;
        case ImportProductionCut::electron:
            return MaterialParams::ProductionCutId::electron;
        case ImportProductionCut::positron:
            return MaterialParams::ProductionCutId::positron;
        case ImportProductionCut::proton:
            return MaterialParams::ProductionCutId::proton;
    }
    CELER_ASSERT_UNREACHABLE();
}
*/
//---------------------------------------------------------------------------//
} // namespace celeritas
