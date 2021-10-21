//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geant-exporter-cat.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "base/Join.hh"
#include "base/Range.hh"
#include "comm/Communicator.hh"
#include "comm/Logger.hh"
#include "comm/ScopedMpiInit.hh"
#include "physics/base/ParticleData.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/CutoffParams.hh"
#include "physics/material/MaterialParams.hh"
#include "io/RootImporter.hh"
#include "io/ImportData.hh"

using namespace celeritas;
using std::cout;
using std::endl;
using std::setprecision;
using std::setw;

//---------------------------------------------------------------------------//
/*!
 * Print particle properties.
 */
void print_particles(const ParticleParams& particles)
{
    CELER_LOG(info) << "Loaded " << particles.size() << " particles";

    cout << R"gfm(
# Particles

| Name              | PDG Code    | Mass [MeV] | Charge [e] | Decay [1/s] |
| ----------------- | ----------- | ---------- | ---------- | ----------- |
)gfm";

    for (auto particle_id : range(ParticleId{particles.size()}))
    {
        const auto& p = particles.get(particle_id);

        // clang-format off
        cout << "| "
             << setw(17) << std::left << particles.id_to_label(particle_id) << " | "
             << setw(11) << particles.id_to_pdg(particle_id).get() << " | "
             << setw(10) << setprecision(6) << p.mass().value() << " | "
             << setw(10) << setprecision(3) << p.charge().value() << " | "
             << setw(11) << setprecision(3) << p.decay_constant()
             << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print element properties.
 *
 * TODO: Use element params directly.
 */
void print_elements(std::vector<ImportElement>& elements)
{
    CELER_LOG(info) << "Loaded " << elements.size() << " elements";
    cout << R"gfm(
# Elements

| Element ID | Name | Atomic number | Mass (AMU) |
| ---------- | ---- | ------------- | ---------- |
)gfm";

    for (unsigned int element_id : range(elements.size()))
    {
        const auto& element = elements.at(element_id);
        // clang-format off
        cout << "| "
             << setw(10) << std::left << element_id << " | "
             << setw(4) << element.name << " | "
             << setw(13) << element.atomic_number << " | "
             << setw(10) << element.atomic_mass << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print material properties.
 *
 * TODO: Use material and cutoff params directly.
 */
void print_materials(std::vector<ImportMaterial>& materials,
                     std::vector<ImportElement>&  elements,
                     const ParticleParams&        particles)
{
    CELER_LOG(info) << "Loaded " << materials.size() << " materials";
    cout << R"gfm(
# Materials

| Material ID | Name                            | Composition                     |
| ----------- | ------------------------------- | ------------------------------- |
)gfm";

    for (unsigned int material_id : range(materials.size()))
    {
        const auto& material = materials.at(material_id);

        cout << "| " << setw(11) << material_id << " | " << setw(31)
             << material.name << " | " << setw(31)
             << to_string(
                    join(material.elements.begin(),
                         material.elements.end(),
                         ", ",
                         [&](const auto& mat_el_comp) {
                             return elements.at(mat_el_comp.element_id).name;
                         }))
             << " |\n";
    }
    cout << endl;

    //// PRINT CUTOFF LIST ///

    cout << R"gfm(
## Cutoffs per material

| Material ID | Name                            | Cutoffs [MeV, cm]               |
| ----------- | ------------------------------- | ------------------------------- |
)gfm";

    std::map<int, std::string> pdg_label;
    for (auto particle_id : range(ParticleId{particles.size()}))
    {
        const int   pdg   = particles.id_to_pdg(particle_id).get();
        const auto& label = particles.id_to_label(particle_id);
        pdg_label.insert({pdg, label});
    }

    for (unsigned int material_id : range(materials.size()))
    {
        bool        is_first_line = true;
        const auto& material      = materials.at(material_id);

        for (const auto& cutoff_key : material.pdg_cutoffs)
        {
            auto iter = pdg_label.find(cutoff_key.first);
            if (iter == pdg_label.end())
                continue;

            const std::string label = iter->second;
            const std::string str_cuts
                = label + ": " + std::to_string(cutoff_key.second.energy)
                  + ", " + std::to_string(cutoff_key.second.range);

            if (is_first_line)
            {
                cout << "| " << setw(11) << material_id << " | " << setw(31)
                     << material.name << " | " << setw(31) << str_cuts
                     << " |\n";
                is_first_line = false;
            }
            else
            {
                cout << "|             |                                 | "
                     << setw(31) << str_cuts << " |\n";
            }
        }
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print process information.
 */
void print_process(const ImportProcess& proc, const ParticleParams& particles)
{
    if (proc.process_type != ImportProcessType::electromagnetic)
    {
        CELER_LOG(warning) << "Skipping non-EM process "
                           << to_cstring(proc.process_class)
                           << " for particle  " << proc.particle_pdg;
    }

    cout << "## "
         << particles.id_to_label(particles.find(PDGNumber{proc.particle_pdg}))
         << " " << to_cstring(proc.process_class) << "\n\n";

    cout << "Models: "
         << join(proc.models.begin(),
                 proc.models.end(),
                 ", ",
                 [](ImportModelClass im) { return to_cstring(im); })
         << "\n";

    for (const auto& table : proc.tables)
    {
        cout << "\n------\n\n" << to_cstring(table.table_type) << ":\n\n";

        cout << "| Type          | Size  | Endpoints ("
             << to_cstring(table.x_units) << ", " << to_cstring(table.y_units)
             << ") |\n"
             << "| ------------- | ----- | "
                "------------------------------------------------------------ "
                "|\n";

        for (const auto& physvec : table.physics_vectors)
        {
            cout << "| " << setw(13) << std::left
                 << to_cstring(physvec.vector_type) << " | " << setw(5)
                 << physvec.x.size() << " | (" << setprecision(3) << setw(12)
                 << physvec.x.front() << ", " << setprecision(3) << setw(12)
                 << physvec.y.front() << ") -> (" << setprecision(3)
                 << setw(12) << physvec.x.back() << ", " << setprecision(3)
                 << setw(12) << physvec.y.back() << ") |\n";
        }
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print stored data for all available processes.
 */
void print_processes(const std::vector<ImportProcess>& processes,
                     const ParticleParams&             particles)
{
    CELER_LOG(info) << "Loaded " << processes.size() << " processes";

    // Print summary
    cout << R"gfm(
# Processes

| Process        | Particle      | Models                    | Tables                   |
| -------------- | ------------- | ------------------------- | ------------------------ |
)gfm";
    for (const ImportProcess& proc : processes)
    {
        auto pdef_id = particles.find(PDGNumber{proc.particle_pdg});
        CELER_ASSERT(pdef_id);

        cout << "| " << setw(14) << to_cstring(proc.process_class) << " | "
             << setw(13) << particles.id_to_label(pdef_id) << " | " << setw(25)
             << to_string(
                    join(proc.models.begin(),
                         proc.models.end(),
                         ", ",
                         [](ImportModelClass im) { return to_cstring(im); }))
             << " | " << setw(24)
             << to_string(join(proc.tables.begin(),
                               proc.tables.end(),
                               ", ",
                               [](const ImportPhysicsTable& tab) {
                                   return to_cstring(tab.table_type);
                               }))
             << " |\n";
    }
    cout << endl;

    // Print details
    for (const ImportProcess& proc : processes)
    {
        print_process(proc, particles);
    }
}
//---------------------------------------------------------------------------//
/*!
 * Print volume properties.
 *
 * TODO: Use a volume params directly when avaliable.
 */
void print_volumes(std::vector<ImportVolume>&   volumes,
                   std::vector<ImportMaterial>& materials)
{
    CELER_LOG(info) << "Loaded " << volumes.size() << " volumes";
    cout << R"gfm(
# Volumes

| Volume ID | Material ID | Solid Name                           | Material Name               |
| --------- | ----------- | ------------------------------------ | --------------------------- |
)gfm";

    for (unsigned int volume_id : range(volumes.size()))
    {
        const auto& volume   = volumes.at(volume_id);
        const auto& material = materials.at(volume.material_id);

        // clang-format off
        cout << "| "
             << setw(9) << std::left << volume_id << " | "
             << setw(11) << volume.material_id << " | "
             << setw(36) << volume.solid_name << " | "
             << setw(27) << material.name << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Dump the contents of a ROOT file writen by geant-exporter.
 */
int main(int argc, char* argv[])
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && Communicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    if (argc != 2)
    {
        // If number of arguments is incorrect, print help
        cout << "Usage: " << argv[0] << " output.root" << endl;
        return 2;
    }

    ImportData data;
    try
    {
        RootImporter import(argv[1]);
        data = import();
    }
    catch (const DebugError& e)
    {
        CELER_LOG(critical) << "Exception while reading ROOT file '" << argv[1]
                            << "': " << e.what();
        return EXIT_FAILURE;
    }

    CELER_LOG(info) << "Successfully loaded ROOT file '" << argv[1] << "'";

    const auto particle_params = ParticleParams::from_import(data);

    print_particles(*particle_params);
    print_elements(data.elements);
    print_materials(data.materials, data.elements, *particle_params);
    print_processes(data.processes, *particle_params);
    print_volumes(data.volumes, data.materials);

    return EXIT_SUCCESS;
}
