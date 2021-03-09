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
#include "physics/base/ParticleInterface.hh"
#include "io/RootImporter.hh"
#include "io/GdmlGeometryMap.hh"

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

    cout << R"gfm(# Particles

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
 * Print physics properties.
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
 * Print physics properties.
 */
void print_processes(const std::vector<ImportProcess>& processes,
                     const ParticleParams&             particles)
{
    CELER_LOG(info) << "Loaded " << processes.size() << " processes";

    // Print summary
    cout << "# Processes\n"
         << R"gfm(
| Process        | Particle      | Models                   | Tables                   |
| -------------- | ------------- | ------------------------ | ------------------------ |
)gfm";
    for (const ImportProcess& proc : processes)
    {
        auto pdef_id = particles.find(PDGNumber{proc.particle_pdg});
        CELER_ASSERT(pdef_id);

        cout << "| " << setw(14) << to_cstring(proc.process_class) << " | "
             << setw(13) << particles.id_to_label(pdef_id) << " | " << setw(24)
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
    cout << "|\n\n";

    // Print details
    for (const ImportProcess& proc : processes)
    {
        print_process(proc, particles);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print GDML properties.
 *
 * TODO: add a print_materials to use material params directly.
 */
void print_geometry(const GdmlGeometryMap& geometry)
{
    //// PRINT ELEMENT LIST ////

    const auto& element_map = geometry.elemid_to_element_map();

    CELER_LOG(info) << "Loaded " << element_map.size() << " elements";
    cout << R"gfm(# GDML properties

## Elements

| Element ID | Name | Atomic number | Mass (AMU) |
| ---------- | ---- | ------------- | ---------- |
)gfm";

    for (const auto& el_key : element_map)
    {
        const auto& element = geometry.get_element(el_key.first);
        // clang-format off
        cout << "| "
             << setw(10) << std::left << el_key.first << " | "
             << setw(4) << element.name << " | "
             << setw(13) << element.atomic_number << " | "
             << setw(10) << element.atomic_mass << " |\n";
        // clang-format on
    }

    //// PRINT MATERIAL LIST ///

    const auto& material_map = geometry.matid_to_material_map();

    CELER_LOG(info) << "Loaded " << material_map.size() << " materials";
    cout << R"gfm(
## Materials

| Material ID | Name                            | Composition                     |
| ----------- | ------------------------------- | ------------------------------- |
)gfm";

    for (const auto& mat_key : material_map)
    {
        const auto& material = geometry.get_material(mat_key.first);
        cout << "| " << setw(11) << mat_key.first << " | " << setw(31)
             << material.name << " | " << setw(31)
             << to_string(join(material.elements_fractions.begin(),
                               material.elements_fractions.end(),
                               ", ",
                               [&geometry](const auto& key) {
                                   return geometry.get_element(key.first).name;
                               }))
             << " |\n";
    }
    cout << endl;

    //// PRINT VOLUME AND MATERIAL LIST ////

    const auto& volume_material_map = geometry.volid_to_matid_map();

    CELER_LOG(info) << "Loaded " << volume_material_map.size() << " volumes";
    cout << R"gfm(
## Volumes

| Volume ID | Material ID | Solid Name                           | Material Name               |
| --------- | ----------- | ------------------------------------ | --------------------------- |
)gfm";

    for (const auto& key_value : volume_material_map)
    {
        auto volid    = key_value.first;
        auto matid    = key_value.second;
        auto volume   = geometry.get_volume(volid);
        auto material = geometry.get_material(matid);

        // clang-format off
        cout << "| "
             << setw(9) << std::left << volid << " | "
             << setw(11) << matid << " | "
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

    RootImporter::result_type data;
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

    print_particles(*data.particle_params);
    print_processes(data.processes, *data.particle_params);
    print_geometry(*data.geometry);

    return EXIT_SUCCESS;
}
