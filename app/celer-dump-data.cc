//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/io/Label.hh"
#include "corecel/io/Logger.hh"
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

#include "CliUtils.hh"

using std::cout;
using std::endl;
using std::fixed;
using std::scientific;
using std::setprecision;
using std::setw;

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

| Name              | PDG Code    | Mass [MeV] | Charge [e] | Decay [1/time] |
| ----------------- | ----------- | ---------- | ---------- | -------------- |
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
             << setw(14) << setprecision(3) << p.decay_constant()
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

    for (auto element_id : range(elements.size()))
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

| Isotope ID/Name   | Atomic number | Atomic mass number | Nuclear mass [MeV] |
| ----------------- | ------------- | ------------------ | ------------------ |
)gfm";

    for (auto isotope_id : range(isotopes.size()))
    {
        auto const& isotope = isotopes[isotope_id];
        // clang-format off
        cout << "| "
             << setw(4) << std::right << isotope_id << ": "
             << setw(11) << std::left << isotope.name << " | "
             << setw(13) << isotope.atomic_number << " | "
             << setw(18) << isotope.atomic_mass_number << " | "
             << setw(18) << isotope.nuclear_mass << " |\n";
        // clang-format on
    }
    cout << endl;

    cout << R"gfm(
## Binding energy [MeV]

| Isotope ID/Name   | Binding energy | Proton loss | Neutron loss |
| ----------------- | -------------- | ----------- | ------------ |
)gfm";

    for (auto isotope_id : range(isotopes.size()))
    {
        auto const& isotope = isotopes[isotope_id];
        // clang-format off
        cout << "| "
             << setw(4) << std::right << isotope_id << ": "
             << setw(11) << std::left << isotope.name << " | "
             << setw(14) << isotope.binding_energy << " | "
             << setw(11) << isotope.proton_loss_energy << " | "
             << setw(12) << isotope.neutron_loss_energy << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print true material properties.
 */
void print_geo_materials(std::vector<ImportGeoMaterial> const& materials,
                         std::vector<ImportElement> const& elements)
{
    CELER_LOG(info) << "Loaded " << materials.size() << " materials";
    cout << R"gfm(
# Geometry materials

| ID/Name                             | Composition                              |
| ----------------------------------- | ---------------------------------------- |
)gfm";

    for (auto material_id : range(materials.size()))
    {
        auto const& material = materials[material_id];

        // clang-format off
        cout << "| "
             << setw(4) << std::right << material_id << ": "
             << setw(29) << std::left << material.name
             << " | "
             << setw(40) << to_string(join(
                    material.elements.begin(),
                    material.elements.end(),
                    ", ",
                    [&](auto const& mat_el_comp) {
                        CELER_ASSERT(mat_el_comp.element_id < elements.size());
                        return elements[mat_el_comp.element_id].name;
                    }))
             << " |\n";
        // clang-format on
    }
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print physics-modified material properties.
 */
void print_phys_materials(std::vector<ImportPhysMaterial> const& materials,
                          std::vector<ImportGeoMaterial> const& geo_materials,
                          ParticleParams const& particles)
{
    //// PRINT CUTOFF LIST ///

    cout << R"gfm(
# Physical material secondary production cutoff

| Material ID/Name                | Particle  | Energy [MeV] | Range [len] |
| ------------------------------- | --------- | ------------ | ----------- |
)gfm";

    for (auto material_id : range(materials.size()))
    {
        bool is_first_line = true;
        auto const& material = materials[material_id];

        for (auto const& [pdg, cuts] : material.pdg_cutoffs)
        {
            if (is_first_line)
            {
                CELER_VALIDATE(material.geo_material_id < geo_materials.size(),
                               << "geo material ID " << material.geo_material_id
                               << " out of range for physics material ID "
                               << material_id);
                ImportGeoMaterial const& geo
                    = geo_materials[material.geo_material_id];
                // clang-format off
                cout << "| " << setw(4) << std::right << material_id
                     << ": " << setw(25) << std::left << geo.name;
                // clang-format on
                is_first_line = false;
            }
            else
            {
                cout << "| " << setw(4) << ' ' << "  " << setw(25) << ' ';
            }

            auto pdef_id = particles.find(PDGNumber{pdg});
            CELER_ASSERT(pdef_id);
            // clang-format off
            cout << " | " << setw(9) << particles.id_to_label(pdef_id)
                 << " | " << setw(12) << cuts.energy
                 << " | " << setw(11) << cuts.range
                 << " |\n";
            // clang-format on
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

    cout << "| Size  | Endpoints (" << to_cstring(table.x_units) << ", "
         << to_cstring(table.y_units) << ") |"
         << R"gfm(
| ----- | ------------------------------------------------------------ |
)gfm";

    for (auto const& physvec : table.physics_vectors)
    {
        cout << "| " << setw(5) << physvec.x.size() << " | ("
             << setprecision(3) << setw(12) << physvec.x.front() << ", "
             << setprecision(3) << setw(12) << physvec.y.front() << ") -> ("
             << setprecision(3) << setw(12) << physvec.x.back() << ", "
             << setprecision(3) << setw(12) << physvec.y.back() << ") |\n";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print process information.
 */
void print_process(ImportProcess const& proc,
                   std::vector<ImportGeoMaterial> const& geo_materials,
                   std::vector<ImportPhysMaterial> const& phys_materials,
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
        cout << "### Model: " << to_cstring(model.model_class) << "\n";
        cout << R"gfm(
| Parameter            | Value     | Units |
| -------------------- | --------- | ----- |
)gfm";
        cout << "| " << setw(20) << std::left << "Low energy limit" << " | "
             << setw(9) << setprecision(3) << model.low_energy_limit << " | "
             << setw(5) << "MeV" << " |\n";
        cout << "| " << setw(20) << std::left << "High energy limit" << " | "
             << setw(9) << setprecision(3) << model.high_energy_limit << " | "
             << setw(5) << "MeV" << " |\n";
        cout << R"gfm(
Energy grids per material:

| Material             | Size  | Endpoints (MeV)              |
| -------------------- | ----- | ---------------------------- |
)gfm";

        for (auto m : range(model.materials.size()))
        {
            auto const& energy = model.materials[m].energy;
            CELER_ASSERT(!energy.empty());
            auto const& geo_mat
                = geo_materials[phys_materials[m].geo_material_id];
            cout << "| " << setw(20) << std::left << geo_mat.name << " | "
                 << setw(5) << energy.size() << " | " << setprecision(3)
                 << setw(12) << setprecision(3) << setw(12) << energy.front()
                 << " -> " << setprecision(3) << setw(12) << energy.back()
                 << " |\n";
        }
        cout << "\n";

        if (std::all_of(model.materials.begin(),
                        model.materials.end(),
                        [](ImportModelMaterial const& imm) {
                            return imm.micro_xs.empty();
                        }))
        {
            cout << "**No microscopic cross sections**\n\n";
            continue;
        }

        cout << R"gfm(
Microscopic cross sections:

| Material             | Element       | Endpoints (bn) |
| -------------------- | ------------- | ---------------------------- |
)gfm";

        for (auto m : range(model.materials.size()))
        {
            using units::barn;

            auto const& xs = model.materials[m].micro_xs;
            auto const& geo_mat
                = geo_materials[phys_materials[m].geo_material_id];
            CELER_VALIDATE(xs.size() == geo_mat.elements.size(),
                           << "mismatched cross section/element size: got "
                           << xs.size() << " micros for "
                           << geo_mat.elements.size() << " element components");

            for (auto i : range(xs.size()))
            {
                auto el_id = geo_mat.elements[i].element_id;
                cout << "| " << setw(20) << std::left
                     << (i == 0 ? geo_mat.name : std::string{}) << " | "
                     << setw(13) << std::left << elements[el_id].name << " | "
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
        print_process(proc,
                      data.geo_materials,
                      data.phys_materials,
                      data.elements,
                      particles);
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
        cout << endl;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print region properties.
 */
void print_regions(std::vector<ImportRegion> const& regions)
{
    CELER_LOG(info) << "Loaded " << regions.size() << " regions";
    cout << R"gfm(
# Regions

| Region ID/name                               | FM | PC | UL |
| -------------------------------------------- | -- | -- | -- |
)gfm";

    auto to_yn = [](bool v) { return v ? 'Y' : 'N'; };

    for (auto region_id : range(regions.size()))
    {
        auto const& region = regions[region_id];

        // clang-format off
        cout << "| "
             << setw(4) << std::right << region_id << ": "
             << setw(34) << std::left << region.name
             << " |  " << to_yn(region.field_manager)
             << " |  " << to_yn(region.production_cuts)
             << " |  " << to_yn(region.user_limits)
             << " |\n";
        // clang-format on
    }
    cout << "\nCustomizations: FM = field manager, PC = production cuts, UL = "
            "user limits\n";
    cout << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print volume properties.
 */
void print_volumes(std::vector<ImportVolume> const& volumes,
                   std::vector<ImportGeoMaterial> const& geo_materials,
                   std::vector<ImportRegion> const& regions)
{
    CELER_LOG(info) << "Loaded " << volumes.size() << " volumes";
    cout << R"gfm(
# Volumes

| Volume ID/name                                     | Phys ID | Region ID/name    | Geo material ID/name  |
| -------------------------------------------------- | ------- | ----------------- | --------------------- |
)gfm";

    for (auto volume_id : range(volumes.size()))
    {
        auto const& volume = volumes[volume_id];
        if (!volume)
        {
            continue;
        }
        // clang-format off
        cout << "| "
             << setw(5) << std::right << volume_id << ": "
             << setw(43) << std::left << volume.name
             << " | " << setw(7);
        // clang-format on
        if (volume.phys_material_id != ImportVolume::unspecified)
        {
            cout << volume.phys_material_id;
        }
        else
        {
            cout << "---";
        }

        if (volume.region_id != ImportVolume::unspecified)
        {
            CELER_VALIDATE(
                static_cast<std::size_t>(volume.region_id) < regions.size(),
                << "region ID " << volume.region_id << " is out of range");
            auto const& region = regions[volume.region_id];
            auto region_name = region.name;

            cout << " | " << setw(3) << std::right << volume.region_id << ": "
                 << setw(12) << std::left << region_name;
        }
        else
        {
            cout << " | " << setw(17) << "---";
        }

        CELER_VALIDATE(static_cast<std::size_t>(volume.geo_material_id)
                           < geo_materials.size(),
                       << "geo material ID " << volume.geo_material_id
                       << " is out of range");
        auto const& geo_mat = geo_materials[volume.geo_material_id];

        cout << " | " << setw(4) << std::right << volume.geo_material_id
             << ": " << setw(7) << std::left << geo_mat.name << " |\n";
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
    "| " << setw(24) << #NAME << " | " << setw(7) << em_params.NAME << " |\n"
#define PEP_STREAM_ENUM(NAME)                     \
    "| " << setw(24) << #NAME << " | " << setw(7) \
         << to_cstring(em_params.NAME) << " |\n"
#define PEP_STREAM_BOOL(NAME)                     \
    "| " << setw(24) << #NAME << " | " << setw(7) \
         << (em_params.NAME ? "true" : "false") << " |\n"
    cout << R"gfm(
# EM parameters

| EM parameter             | Value   |
| ------------------------ | ------- |
)gfm";
    cout << PEP_STREAM_BOOL(energy_loss_fluct) << PEP_STREAM_BOOL(lpm)
         << PEP_STREAM_BOOL(integral_approach)
         << PEP_STREAM_PARAM(linear_loss_limit)
         << PEP_STREAM_PARAM(lowest_electron_energy)
         << PEP_STREAM_PARAM(lowest_muhad_energy) << PEP_STREAM_BOOL(auger)
         << PEP_STREAM_ENUM(msc_step_algorithm)
         << PEP_STREAM_ENUM(msc_muhad_step_algorithm)
         << PEP_STREAM_PARAM(msc_displaced)
         << PEP_STREAM_PARAM(msc_muhad_displaced)
         << PEP_STREAM_PARAM(msc_range_factor)
         << PEP_STREAM_PARAM(msc_muhad_range_factor)
         << PEP_STREAM_PARAM(msc_safety_factor)
         << PEP_STREAM_PARAM(msc_lambda_limit)
         << PEP_STREAM_PARAM(msc_theta_limit) << PEP_STREAM_BOOL(apply_cuts)
         << PEP_STREAM_PARAM(screening_factor)
         << PEP_STREAM_PARAM(angle_limit_factor) << endl;
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
        auto const& par = particles.id_to_label(pid);
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

| Atomic number | Endpoints (x, y, value [mb])                               |
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
 * Print muon pair production sampling table.
 */
void print_mupp_data(ImportMuPairProductionTable const& mupp_data)
{
    if (!mupp_data)
    {
        CELER_LOG(info) << "Muon pair production sampling table not available";
        return;
    }

    CELER_LOG(info) << "Loaded muon pair production sampling table with size "
                    << mupp_data.physics_vectors.size();

    cout << R"gfm(
# Muon pair production sampling table

| Atomic number | Endpoints (x, y, value)                                     |
| ------------- | ----------------------------------------------------------- |
)gfm";

    for (auto i : range(mupp_data.atomic_number.size()))
    {
        auto z = mupp_data.atomic_number[i];
        auto const& pv = mupp_data.physics_vectors[i];

        cout << "| " << setw(13) << z << " | (" << setprecision(3) << setw(7)
             << pv.x.front() << ", " << setprecision(3) << setw(7)
             << pv.y.front() << ", " << setprecision(3) << setw(7)
             << pv.value.front() << ") -> (" << setprecision(3) << setw(7)
             << pv.x.back() << ", " << setprecision(3) << setw(7)
             << pv.y.back() << ", " << setprecision(3) << setw(8)
             << pv.value.back() << ") |\n";
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
#define POM_STREAM_SCALAR_COMP(ID, STRUCT, NAME, UNITS, COMP)             \
    "| " << setw(11) << ID << " | " << setw(20) << #NAME << COMP << " | " \
         << setw(15) << to_cstring(UNITS) << " | " << setprecision(3)     \
         << setw(9) << STRUCT.NAME << " | " << setw(52) << ""             \
         << " | " << setw(7) << ""                                        \
         << " |\n"
#define POM_STREAM_SCALAR(ID, STRUCT, NAME, UNITS) \
    POM_STREAM_SCALAR_COMP(ID, STRUCT, NAME, UNITS, "      ")
#define POM_STREAM_VECTOR(ID, STRUCT, NAME, UNITS)                            \
    "| " << setw(11) << ID << " | " << setw(26) << #NAME << " | " << setw(15) \
         << to_cstring(UNITS) << " | " << setw(9) << ""                       \
         << " | (" << setprecision(3) << setw(10) << STRUCT.NAME.x.front()    \
         << ", " << setprecision(3) << setw(10) << STRUCT.NAME.y.front()      \
         << ") -> (" << setprecision(3) << setw(10) << STRUCT.NAME.x.back()   \
         << ", " << setprecision(3) << setw(10) << STRUCT.NAME.y.back()       \
         << ") | " << setw(7) << STRUCT.NAME.x.size() << " |\n";

/*!
 * Helper class for printing imported optical models.
 */
class OpticalMfpHelper
{
  public:
    //!@{
    //! \name Type aliases
    using VecModels = std::vector<ImportOpticalModel>;
    //!@}

    // MFP to print for the model
    struct MfpPrinter
    {
        inp::Grid const& mfp;
    };

    //! Construct helper for given model class out of the models
    OpticalMfpHelper(std::vector<ImportOpticalModel> const& models,
                     optical::ImportModelClass imc)
    {
        auto iter
            = std::find_if(models.begin(), models.end(), [imc](auto const& m) {
                  return m.model_class == imc;
              });
        if (iter != models.end())
        {
            mfps_ = &iter->mfp_table;
        }
    }

    //! Print the MFP if the model exists and has the given material
    void print_mfp(std::size_t mid) const
    {
        if (mfps_ && mid < mfps_->size())
        {
            MfpPrinter printer{(*mfps_)[mid]};
            cout << POM_STREAM_VECTOR(mid, printer, mfp, ImportUnits::len);
        }
    }

  private:
    std::vector<inp::Grid> const* mfps_{nullptr};
};

/*!
 * Print optical material properties map.
 */
void print_optical_materials(std::vector<ImportOpticalModel> const& io_models,
                             std::vector<ImportOpticalMaterial> const& io_mats)
{
    if (io_mats.empty())
    {
        CELER_LOG(info) << "Optical material data not available";
        return;
    }

    if (io_models.empty())
    {
        CELER_LOG(info) << "Optical model data not available";
    }

    CELER_LOG(info) << "Loaded optical material data map with size "
                    << io_mats.size();

    static char const header[] = R"gfm(

| Material ID | Property                   | Units           | Scalar    | Vector endpoints (MeV, value)                        | Size    |
| ----------- | -------------------------- | --------------- | --------- | ---------------------------------------------------- | ------- |
)gfm";

    using IU = ImportUnits;

    cout << "\n# Optical properties\n";
    cout << "\n## Common properties";
    cout << header;
    for (auto mid : range(io_mats.size()))
    {
        auto const& prop = io_mats[mid].properties;
        cout << POM_STREAM_VECTOR(mid, prop, refractive_index, IU::unitless);
    }

    cout << "\n## Scintillation";
    cout << header;
    static char const* comp_str[] = {"(fast)", " (mid)", "(slow)"};
    for (auto mid : range(io_mats.size()))
    {
        auto const& scint = io_mats[mid].scintillation;
        if (!scint)
        {
            continue;
        }
        cout << POM_STREAM_SCALAR(
            mid, scint, material.yield_per_energy, IU::inv_mev);
        cout << POM_STREAM_SCALAR(mid, scint, resolution_scale, IU::unitless);
        for (auto i : range(scint.material.components.size()))
        {
            auto const& comp = scint.material.components[i];
            cout << POM_STREAM_SCALAR_COMP(
                mid, comp, yield_frac, IU::inv_mev, comp_str[i]);
            cout << POM_STREAM_SCALAR_COMP(
                mid, comp, lambda_mean, IU::len, comp_str[i]);
            cout << POM_STREAM_SCALAR_COMP(
                mid, comp, lambda_sigma, IU::len, comp_str[i]);
            cout << POM_STREAM_SCALAR_COMP(
                mid, comp, rise_time, IU::time, comp_str[i]);
            cout << POM_STREAM_SCALAR_COMP(
                mid, comp, fall_time, IU::time, comp_str[i]);
        }
    }

    {
        OpticalMfpHelper rayleigh_model(io_models,
                                        optical::ImportModelClass::rayleigh);

        cout << "\n## Rayleigh";
        cout << header;
        for (auto mid : range(io_mats.size()))
        {
            rayleigh_model.print_mfp(mid);

            auto const& rayl = io_mats[mid].rayleigh;
            if (!rayl)
            {
                continue;
            }

            cout << POM_STREAM_SCALAR(mid, rayl, scale_factor, IU::unitless);
            cout << POM_STREAM_SCALAR(
                mid, rayl, compressibility, IU::len_time_sq_per_mass);
        }
    }

    {
        OpticalMfpHelper absorption_helper(
            io_models, optical::ImportModelClass::absorption);

        cout << "\n## Absorption";
        cout << header;
        for (auto mid : range(io_mats.size()))
        {
            absorption_helper.print_mfp(mid);
        }
        cout << endl;
    }

    {
        OpticalMfpHelper wls_helper(io_models, optical::ImportModelClass::wls);

        cout << "\n## WLS";
        cout << header;
        for (auto mid : range(io_mats.size()))
        {
            wls_helper.print_mfp(mid);

            auto const& wls = io_mats[mid].wls;
            if (!wls)
            {
                continue;
            }

            cout << POM_STREAM_SCALAR(mid, wls, mean_num_photons, IU::unitless);
            cout << POM_STREAM_SCALAR(mid, wls, time_constant, IU::time);
            cout << POM_STREAM_VECTOR(mid, wls, component, IU::unitless);
        }
        cout << endl;
    }
}

#undef PEP_STREAM_SCALAR
#undef PEP_STREAM_VECTOR

//---------------------------------------------------------------------------//
}  // namespace

void run(std::string const& filename)
{
    ScopedRootErrorHandler scoped_root_error;
    RootImporter import(filename);
    auto data = import();
    scoped_root_error.throw_if_errors();

    cout << "Contents of `" << filename << "` (" << data.units
         << " unit system)\n\n"
            "-----\n\n";

    auto const&& particle_params = ParticleParams::from_import(data);

    print_elements(data.elements, data.isotopes);
    print_isotopes(data.isotopes);

    print_geo_materials(data.geo_materials, data.elements);
    print_phys_materials(
        data.phys_materials, data.geo_materials, *particle_params);
    print_optical_materials(data.optical_models, data.optical_materials);

    print_regions(data.regions);
    print_volumes(data.volumes, data.geo_materials, data.regions);

    print_particles(*particle_params);
    print_processes(data, *particle_params);
    print_msc_models(data, *particle_params);

    print_sb_data(data.sb_data);
    print_livermore_pe_data(data.livermore_pe_data);
    print_mupp_data(data.mu_pair_production_data);
    print_atomic_relaxation_data(data.atomic_relaxation_data);

    print_em_params(data.em_params);
    print_trans_params(data.trans_params, *particle_params);
    // TODO: print optical params?
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])  // NOLINT(bugprone-exception-escape)
{
    using namespace celeritas::app;

    // Set up MPI
    celeritas::ScopedMpiInit scoped_mpi(&argc, &argv);
    if (scoped_mpi.is_world_multiprocess())
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    // Set up app
    auto& cli = cli_app();
    cli.description("Dump Geant4 data to ROOT");

    std::string filename;
    cli.add_option("filename", filename, "Input ROOT file")
        ->required()
        ->check(CLI::ExistingFile);

    // Parse and run
    CELER_CLI11_PARSE(argc, argv);
    return run_safely(run, filename);
}
