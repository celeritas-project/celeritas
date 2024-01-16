//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-dump-data.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/detail/Joined.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

using std::cout;
using std::endl;
using std::fixed;
using std::scientific;
using std::setprecision;
using std::setw;
using std::stringstream;

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Print particle properties.
 */
void print_particles(ParticleParams const& particles)
{
    CELER_LOG(info) << "Loaded " << particles.size() << " particles";

    cout << R"gfm(
# Particles

| Name              | PDG Code    | Mass [MeV] | Charge [e] | Decay [1/s] |
| ----------------- | ----------- | ---------- | ---------- | ----------- |
)gfm";

    for (auto particle_id : range(ParticleId{particles.size()}))
    {
        auto const& p = particles.get(particle_id);

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
void print_elements(std::vector<ImportElement>& elements,
                    std::vector<ImportIsotope>& isotopes)
{
    CELER_LOG(info) << "Loaded " << elements.size() << " elements";
    cout << R"gfm(
# Elements

| Element ID | Name | Atomic number | Mass (AMU) | Isotopes                                 |
| ---------- | ---- | ------------- | ---------- | ---------------------------------------- |
)gfm";

    for (unsigned int element_id : range(elements.size()))
    {
        auto const& element = elements[element_id];

        auto const labels = to_string(
            join(element.isotopes_fractions.begin(),
                 element.isotopes_fractions.end(),
                 ", ",
                 [&](auto const& key) { return isotopes[key.first].name; }));

        // clang-format off
        cout << "| "
             << setw(10) << std::left << element_id << " | "
             << setw(4)  << element.name << " | "
             << setw(13) << element.atomic_number << " | "
             << setw(10) << element.atomic_mass << " | "
             << setw(40) << labels << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print isotope properties.
 */
void print_isotopes(std::vector<ImportIsotope>& isotopes)
{
    CELER_LOG(info) << "Loaded " << isotopes.size() << " isotopes";
    cout << R"gfm(
# Isotopes

| Isotope ID | Name   | Atomic number | Atomic mass number | Nuclear mass (MeV) |
| ---------- | ------ | ------------- | ------------------ | ------------------ |
)gfm";

    for (unsigned int isotope_id : range(isotopes.size()))
    {
        auto const& isotope = isotopes[isotope_id];
        // clang-format off
        cout << "| "
             << setw(10) << std::left << isotope_id << " | "
             << setw(6) << isotope.name << " | "
             << setw(13) << isotope.atomic_number << " | "
             << setw(18) << isotope.atomic_mass_number << " | "
             << setw(18) << isotope.nuclear_mass << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print material properties.
 */
void print_materials(std::vector<ImportMaterial>& materials,
                     std::vector<ImportElement>& elements,
                     ParticleParams const& particles)
{
    CELER_LOG(info) << "Loaded " << materials.size() << " materials";
    cout << R"gfm(
# Materials

| Material ID | Name                            | Composition                     |
| ----------- | ------------------------------- | ------------------------------- |
)gfm";

    for (unsigned int material_id : range(materials.size()))
    {
        auto const& material = materials[material_id];

        cout << "| " << setw(11) << material_id << " | " << setw(31)
             << material.name << " | " << setw(31)
             << to_string(join(
                    material.elements.begin(),
                    material.elements.end(),
                    ", ",
                    [&](auto const& mat_el_comp) {
                        CELER_ASSERT(mat_el_comp.element_id < elements.size());
                        return elements[mat_el_comp.element_id].name;
                    }))
             << " |\n";
    }
    cout << endl;

    //// PRINT CUTOFF LIST ///

    cout << R"gfm(
## Cutoffs

| Material                   | Particle  | Energy [MeV] | Range [cm] |
| -------------------------- | --------- | ------------ | ---------- |
)gfm";

    for (unsigned int material_id : range(materials.size()))
    {
        bool is_first_line = true;
        auto const& material = materials[material_id];

        for (auto const& [pdg, cuts] : material.pdg_cutoffs)
        {
            if (is_first_line)
            {
                cout << "| " << std::right << setw(4) << material_id << ": "
                     << setw(20) << material.name;
                is_first_line = false;
            }
            else
            {
                cout << "| " << setw(4) << ' ' << "  " << setw(20) << ' ';
            }

            auto pdef_id = particles.find(PDGNumber{pdg});
            CELER_ASSERT(pdef_id);
            cout << " | " << setw(9) << particles.id_to_label(pdef_id) << " | "
                 << setw(12) << cuts.energy << " | " << setw(10) << cuts.range
                 << " |\n";
        }
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print a physics table.
 */
void print_table(ImportPhysicsTable const& table)
{
    cout << to_cstring(table.table_type) << ":\n\n";

    cout << "| Type          | Size  | Endpoints ("
         << to_cstring(table.x_units) << ", " << to_cstring(table.y_units)
         << ") |\n"
         << "| ------------- | ----- | "
            "------------------------------------------------------------ "
            "|\n";

    for (auto const& physvec : table.physics_vectors)
    {
        cout << "| " << setw(13) << std::left
             << to_cstring(physvec.vector_type) << " | " << setw(5)
             << physvec.x.size() << " | (" << setprecision(3) << setw(12)
             << physvec.x.front() << ", " << setprecision(3) << setw(12)
             << physvec.y.front() << ") -> (" << setprecision(3) << setw(12)
             << physvec.x.back() << ", " << setprecision(3) << setw(12)
             << physvec.y.back() << ") |\n";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print process information.
 */
void print_process(ImportProcess const& proc,
                   std::vector<ImportMaterial> const& materials,
                   std::vector<ImportElement> const& elements,
                   ParticleParams const& particles)
{
    auto pdef_id = particles.find(PDGNumber{proc.particle_pdg});
    cout << "## Process: " << to_cstring(proc.process_class) << " ("
         << particles.id_to_label(pdef_id) << ")\n\n";

    if (proc.tables.empty())
    {
        cout << "**No macroscopic cross sections**\n\n";
    }

    for (ImportModel const& model : proc.models)
    {
        cout << "### Model: " << to_cstring(model.model_class)
             << "\n"
                "\n"
                "Energy grids per material: \n\n"
                "| Material             | Size  | Endpoints (MeV)         "
                "    "
                " |\n"
                "| -------------------- | ----- | "
                "---------------------------- |\n";

        for (auto m : range(model.materials.size()))
        {
            auto const& energy = model.materials[m].energy;
            CELER_ASSERT(!energy.empty());
            cout << "| " << setw(20) << std::left << materials[m].name << " | "
                 << setw(5) << energy.size() << " | " << setprecision(3)
                 << setw(12) << setprecision(3) << setw(12) << energy.front()
                 << " -> " << setprecision(3) << setw(12) << energy.back()
                 << " |\n";
        }
        cout << "\n\n";

        if (std::all_of(model.materials.begin(),
                        model.materials.end(),
                        [](ImportModelMaterial const& imm) {
                            return imm.micro_xs.empty();
                        }))
        {
            cout << "**No microscopic cross sections**\n\n";
            continue;
        }

        cout << "Microscopic cross sections:\n\n"
                "| Material             | Element       | Endpoints (bn) "
                "|\n"
                "| -------------------- | ------------- | "
                "---------------------------- |\n";

        for (auto m : range(model.materials.size()))
        {
            using units::barn;

            auto const& xs = model.materials[m].micro_xs;

            for (auto i : range(xs.size()))
            {
                cout << "| " << setw(20) << std::left
                     << (i == 0 ? materials[m].name : std::string{}) << " | "
                     << setw(13) << std::left << elements[i].name << " | "
                     << setprecision(3) << setw(12) << xs[i].front() / barn
                     << " -> " << setprecision(3) << setw(12)
                     << xs[i].back() / barn << " |\n";
            }
        }
        cout << endl;
    }

    if (proc.tables.empty())
    {
        return;
    }

    cout << "### Macroscopic cross-sections\n\n";

    bool is_first{true};
    for (auto const& table : proc.tables)
    {
        if (!is_first)
        {
            cout << "\n------\n\n";
        }
        else
        {
            is_first = false;
        }

        print_table(table);
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print stored data for all available processes.
 */
void print_processes(ImportData const& data, ParticleParams const& particles)
{
    auto const& processes = data.processes;
    CELER_LOG(info) << "Loaded " << processes.size() << " processes";

    // Print summary
    cout << R"gfm(
# Processes

| Process        | Particle      | Models                    | Tables                          |
| -------------- | ------------- | ------------------------- | ------------------------------- |
)gfm";
    for (ImportProcess const& proc : processes)
    {
        auto pdef_id = particles.find(PDGNumber{proc.particle_pdg});
        CELER_ASSERT(pdef_id);

        cout << "| " << setw(14) << to_cstring(proc.process_class) << " | "
             << setw(13) << particles.id_to_label(pdef_id) << " | " << setw(25)
             << to_string(join(proc.models.begin(),
                               proc.models.end(),
                               ", ",
                               [](ImportModel const& im) {
                                   return to_cstring(im.model_class);
                               }))
             << " | " << setw(31)
             << to_string(join(proc.tables.begin(),
                               proc.tables.end(),
                               ", ",
                               [](ImportPhysicsTable const& tab) {
                                   return to_cstring(tab.table_type);
                               }))
             << " |\n";
    }
    cout << endl;

    // Print details
    for (ImportProcess const& proc : processes)
    {
        print_process(proc, data.materials, data.elements, particles);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print stored data for multiple scattering models.
 */
void print_msc_models(ImportData const& data, ParticleParams const& particles)
{
    auto const& models = data.msc_models;
    CELER_LOG(info) << "Loaded " << models.size() << " MSC models";

    cout << "\n"
            "# MSC models\n\n";

    for (ImportMscModel const& m : models)
    {
        auto pdef_id = particles.find(PDGNumber{m.particle_pdg});
        CELER_ASSERT(pdef_id);
        cout << "## " << particles.id_to_label(pdef_id) << " "
             << to_cstring(m.model_class) << "\n\n";

        print_table(m.xs_table);
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print volume properties.
 */
void print_volumes(std::vector<ImportVolume> const& volumes,
                   std::vector<ImportMaterial> const& materials)
{
    CELER_LOG(info) << "Loaded " << volumes.size() << " volumes";
    cout << R"gfm(
# Volumes

| Volume ID | Volume name                          | Material ID | Material Name               |
| --------- | ------------------------------------ | ----------- | --------------------------- |
)gfm";

    for (unsigned int volume_id : range(volumes.size()))
    {
        auto const& volume = volumes[volume_id];
        if (!volume)
        {
            continue;
        }
        CELER_ASSERT(static_cast<std::size_t>(volume.material_id)
                     < materials.size());

        // clang-format off
        cout << "| "
             << setw(9) << std::left << volume_id << " | "
             << setw(36) << volume.name << " | "
             << setw(11) << volume.material_id << " | "
             << setw(27) << materials[volume.material_id].name << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print EM parameters.
 */
void print_em_params(ImportEmParameters const& em_params)
{
    // NOTE: boolalpha doesn't work with setw, see
    // https://timsong-cpp.github.io/lwg-issues/2703
#define PEP_STREAM_PARAM(NAME) \
    "| " << setw(22) << #NAME << " | " << setw(7) << em_params.NAME << " |\n"
#define PEP_STREAM_BOOL(NAME)                     \
    "| " << setw(22) << #NAME << " | " << setw(7) \
         << (em_params.NAME ? "true" : "false") << " |\n"
    cout << R"gfm(
# EM parameters

| EM parameter           | Value   |
| ---------------------- | ------- |
)gfm";
    cout << PEP_STREAM_BOOL(energy_loss_fluct) << PEP_STREAM_BOOL(lpm)
         << PEP_STREAM_BOOL(integral_approach)
         << PEP_STREAM_PARAM(linear_loss_limit)
         << PEP_STREAM_PARAM(lowest_electron_energy) << PEP_STREAM_BOOL(auger)
         << PEP_STREAM_PARAM(msc_range_factor)
         << PEP_STREAM_PARAM(msc_safety_factor)
         << PEP_STREAM_PARAM(msc_lambda_limit) << PEP_STREAM_BOOL(apply_cuts)
         << PEP_STREAM_PARAM(screening_factor) << endl;
#undef PEP_STREAM_PARAM
#undef PEP_STREAM_BOOL
}

//---------------------------------------------------------------------------//
/*!
 * Print transportation parameters.
 */
void print_trans_params(ImportTransParameters const& trans_params,
                        ParticleParams const& particles)
{
#define PEP_STREAM_PARAM(NAME)                          \
    "| " << setw(24) << #NAME << " | " << setw(9) << "" \
         << " | " << setw(7) << trans_params.NAME << " |\n"
#define PEP_STREAM_PAR_PARAM(NAME, PAR)                                      \
    "| " << setw(24) << #NAME << " | " << setw(9) << PAR << " | " << setw(7) \
         << kv.second.NAME << " |\n"
    cout << R"gfm(
# Transportation parameters

| Transportation parameter | Particle  | Value   |
| ------------------------ | --------- | ------- |
)gfm";
    cout << PEP_STREAM_PARAM(max_substeps);
    for (auto const& kv : trans_params.looping)
    {
        auto pid = particles.find(PDGNumber{kv.first});
        auto par = particles.id_to_label(pid);
        cout << PEP_STREAM_PAR_PARAM(threshold_trials, par)
             << PEP_STREAM_PAR_PARAM(important_energy, par);
    }
    cout << endl;
#undef PEP_STREAM_PAR_PARAM
#undef PEP_STREAM_PARAM
}

//---------------------------------------------------------------------------//
/*!
 * Print Seltzer-Berger map.
 */
void print_sb_data(ImportData::ImportSBMap const& sb_map)
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

    for (auto const& key : sb_map)
    {
        auto const& table = key.second;

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
void print_livermore_pe_data(ImportData::ImportLivermorePEMap const& lpe_map)
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

    for (auto const& key : lpe_map)
    {
        auto const& ilpe = key.second;

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
    ImportData::ImportAtomicRelaxationMap const& ar_map)
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

    for (auto const& key : ar_map)
    {
        auto const& iar = key.second;

        cout << "| " << setw(13) << key.first << " | " << setw(13)
             << iar.shells.size() << " |\n";
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace app
}  // namespace celeritas

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    using namespace celeritas;
    using namespace celeritas::app;

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
        std::cerr << "usage: " << argv[0] << " {output}.root" << std::endl;
        return 2;
    }

    ImportData data;
    try
    {
        ScopedRootErrorHandler scoped_root_error;
        RootImporter import(argv[1]);
        data = import();
        scoped_root_error.throw_if_errors();
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical)
            << "While processing ROOT data at " << argv[1] << ": " << e.what();

        return EXIT_FAILURE;
    }

    cout << "Contents of `" << argv[1]
         << "`\n\n"
            "-----\n\n";

    auto const&& particle_params = ParticleParams::from_import(data);

    print_particles(*particle_params);
    print_elements(data.elements, data.isotopes);
    print_isotopes(data.isotopes);
    print_materials(data.materials, data.elements, *particle_params);
    print_processes(data, *particle_params);
    print_msc_models(data, *particle_params);
    print_volumes(data.volumes, data.materials);
    print_em_params(data.em_params);
    print_trans_params(data.trans_params, *particle_params);
    print_sb_data(data.sb_data);
    print_livermore_pe_data(data.livermore_pe_data);
    print_atomic_relaxation_data(data.atomic_relaxation_data);

    return EXIT_SUCCESS;
}
