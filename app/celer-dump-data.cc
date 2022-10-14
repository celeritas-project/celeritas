//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-dump-data.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleParams.hh"

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
void print_process(const ImportProcess&               proc,
                   const std::vector<ImportMaterial>& materials,
                   const std::vector<ImportElement>&  elements,
                   const ParticleParams&              particles)
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
         << "\n\n";

    cout << "### Microscopic cross-sections\n\n";

    for (const auto& iter : proc.micro_xs)
    {
        // Print models
        cout << to_cstring(iter.first) << "\n";

        const auto& micro_xs = iter.second;
        for (size_type mat_id = 0; mat_id < micro_xs.size(); mat_id++)
        {
            // Print materials
            cout << "\n- " << materials.at(mat_id).name << "\n\n";
            cout << "| Element       | Size  | Endpoints (MeV, cm^2) |\n"
                 << "| ------------- | ----- | "
                    "-----------------------------------------------------"
                    "------- "
                    "|\n";

            const auto& elem_phys_vectors = micro_xs.at(mat_id);

            for (size_t i : celeritas::range(elem_phys_vectors.size()))
            {
                // Print elements and their physics vectors
                const auto physvec = elem_phys_vectors.at(i);
                cout << "| " << setw(13) << std::left << elements.at(i).name
                     << " | " << setw(5) << physvec.x.size() << " | ("
                     << setprecision(3) << setw(12) << physvec.x.front()
                     << ", " << setprecision(3) << setw(12)
                     << physvec.y.front() << ") -> (" << setprecision(3)
                     << setw(12) << physvec.x.back() << ", " << setprecision(3)
                     << setw(12) << physvec.y.back() << ") |\n";
            }
        }

        cout << "\n------\n\n";
    }

    cout << "### Macroscopic cross-sections\n";

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
void print_processes(const ImportData& data, const ParticleParams& particles)
{
    const auto& processes = data.processes;
    CELER_LOG(info) << "Loaded " << processes.size() << " processes";

    // Print summary
    cout << R"gfm(
# Processes

| Process        | Particle      | Models                    | Tables                          |
| -------------- | ------------- | ------------------------- | ------------------------------- |
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
             << " | " << setw(31)
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
        print_process(proc, data.materials, data.elements, particles);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print volume properties.
 */
void print_volumes(std::vector<ImportVolume>&   volumes,
                   std::vector<ImportMaterial>& materials)
{
    CELER_LOG(info) << "Loaded " << volumes.size() << " volumes";
    cout << R"gfm(
# Volumes

| Volume ID | Volume name                          | Material ID | Material Name               |
| --------- | ------------------------------------ | ----------- | --------------------------- |
)gfm";

    for (unsigned int volume_id : range(volumes.size()))
    {
        const auto& volume   = volumes.at(volume_id);
        const auto& material = materials.at(volume.material_id);

        // clang-format off
        cout << "| "
             << setw(9) << std::left << volume_id << " | "
             << setw(36) << volume.name << " | "
             << setw(11) << volume.material_id << " | "
             << setw(27) << material.name << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print EM parameters.
 */
void print_em_params(ImportData::ImportEmParamsMap& em_params_map)
{
    if (em_params_map.empty())
    {
        CELER_LOG(info) << "EM Parameters not available";
        return;
    }

    CELER_LOG(info) << "Loaded " << em_params_map.size() << " EM parameters";

    cout << R"gfm(
# EM parameters

| EM parameter       | Value   |
| ------------------ | ------- |
)gfm";

    for (const auto& key : em_params_map)
    {
        cout << "| " << setw(18) << to_cstring(key.first) << " | " << setw(7)
             << key.second << " |\n";
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print Seltzer-Berger map.
 */
void print_sb_data(const ImportData::ImportSBMap& sb_map)
{
    if (sb_map.empty())
    {
        CELER_LOG(info) << "Seltzer-Berger data not available";
        return;
    }

    CELER_LOG(info) << "Loaded " << sb_map.size() << " SB tables";

    cout << R"gfm(
# Seltzer-Berger data

| Atomic number | Endpoints (x, y, value [mb]) |
| ------------- | ---------------------------------------------------------- |
)gfm";

    for (const auto& key : sb_map)
    {
        const auto& table = key.second;

        cout << "| " << setw(13) << key.first << " | (" << setprecision(3)
             << setw(7) << table.x.front() << ", " << setprecision(3)
             << setw(7) << table.y.front() << ", " << setprecision(3)
             << setw(7) << table.value.front() << ") -> (" << setprecision(3)
             << setw(7) << table.x.back() << ", " << setprecision(3) << setw(7)
             << table.y.back() << ", " << setprecision(3) << setw(7)
             << table.value.back() << ") |\n";
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print Livermore PE map.
 */
void print_livermore_pe_data(const ImportData::ImportLivermorePEMap& lpe_map)
{
    if (lpe_map.empty())
    {
        CELER_LOG(info) << "Livermore PE data not available";
        return;
    }

    CELER_LOG(info) << "Loaded Livermore PE data map with size "
                    << lpe_map.size();

    cout << R"gfm(
# Livermore PE data

| Atomic number | Thresholds (low, high) [MeV] | Subshell size |
| ------------- | ---------------------------- | ------------- |
)gfm";

    for (const auto& key : lpe_map)
    {
        const auto& ilpe = key.second;

        cout << "| " << setw(13) << key.first << " | (" << setprecision(3)
             << setw(12) << ilpe.thresh_lo << ", " << setprecision(3)
             << setw(12) << ilpe.thresh_hi << ") | " << setw(13)
             << ilpe.shells.size() << " |\n";
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print atomic relaxation map.
 */
void print_atomic_relaxation_data(
    const ImportData::ImportAtomicRelaxationMap& ar_map)
{
    if (ar_map.empty())
    {
        CELER_LOG(info) << "Atomic relaxation data not available";
        return;
    }

    CELER_LOG(info) << "Loaded atomic relaxation data map with size "
                    << ar_map.size();

    cout << R"gfm(
# Atomic relaxation data

| Atomic number | Subshell size |
| ------------- | ------------- |
)gfm";

    for (const auto& key : ar_map)
    {
        const auto& iar = key.second;

        cout << "| " << setw(13) << key.first << " | " << setw(13)
             << iar.shells.size() << " |\n";
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Dump the contents of a ROOT file writen by celer-export-geant.
 */
int main(int argc, char* argv[])
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && MpiCommunicator::comm_world().size() > 1)
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
    catch (const RuntimeError& e)
    {
        CELER_LOG(critical) << "Runtime error: " << e.what();
        return EXIT_FAILURE;
    }
    catch (const DebugError& e)
    {
        CELER_LOG(critical) << "Assertion failure: " << e.what();
        return EXIT_FAILURE;
    }

    const auto particle_params = ParticleParams::from_import(data);

    print_particles(*particle_params);
    print_elements(data.elements);
    print_materials(data.materials, data.elements, *particle_params);
    print_processes(data, *particle_params);
    print_volumes(data.volumes, data.materials);
    print_em_params(data.em_params);
    print_sb_data(data.sb_data);
    print_livermore_pe_data(data.livermore_pe_data);
    print_atomic_relaxation_data(data.atomic_relaxation_data);

    return EXIT_SUCCESS;
}
