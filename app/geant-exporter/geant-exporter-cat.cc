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

#include "base/Range.hh"
#include "comm/Communicator.hh"
#include "comm/Logger.hh"
#include "comm/ScopedMpiInit.hh"
#include "physics/base/ParticleDef.hh"
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

-----------------------------------------------------------------------
Name              | PDG Code    | Mass [MeV] | Charge [e] | Decay [1/s]
----------------- | ----------- | ---------- | ---------- | -----------
)gfm";

    for (auto idx : range<ParticleDefId::value_type>(particles.size()))
    {
        ParticleDefId      def_id{idx};
        const ParticleDef& def = particles.get(def_id);

        // clang-format off
        cout << setw(17) << std::left << particles.id_to_label(def_id) << " | "
             << setw(11) << particles.id_to_pdg(def_id).get() << " | "
             << setw(10) << setprecision(6) << def.mass.value() << " | "
             << setw(10) << setprecision(3) << def.charge.value() << " | "
             << setw(11) << setprecision(3) << def.decay_constant
             << '\n';
        // clang-format on
    }
    cout << "-----------------------------------------------------------------"
            "------\n"
         << endl;
}

//---------------------------------------------------------------------------//
/*!
 * Print physics properties.
 */
void print_physics_table(const ImportPhysicsTable& table)
{
    if (table.process_type != ImportProcessType::electromagnetic)
    {
        CELER_LOG(warning)
            << "Skipping non-EM table for particle  " << table.particle.get()
            << ": " << to_cstring(table.process_type) << '.'
            << to_cstring(table.process) << '.' << to_cstring(table.model)
            << ": " << to_cstring(table.table_type);
    }
    cout << "## Particle " << table.particle.get() << ": `"
         << to_cstring(table.process_type) << '.' << to_cstring(table.process)
         << '.' << to_cstring(table.model)
         << "`: " << to_cstring(table.table_type) << " ("
         << to_cstring(table.units) << ")\n\n";

    cout << "Has " << table.physics_vectors.size() << " vectors:\n"
         << R"gfm(
-----------------------------------------------------------------------------------
Type          | Size  | Emin         | Emax         | Vmin         | Vmax
-----------------------------------------------------------------------------------
)gfm";

    for (const auto& physvec : table.physics_vectors)
    {
        // clang-format off
        cout << setw(13) << std::left << to_cstring(physvec.vector_type) << " | "
             << setw(5) << physvec.energy.size() << " | "
             << setw(12) << setprecision(3) << physvec.energy.front() << " | "
             << setw(12) << setprecision(3) << physvec.energy.back() << " | "
             << setw(12) << setprecision(3) << physvec.value.front() << " | "
             << setw(12) << setprecision(3) << physvec.value.back()
             << '\n';
        // clang-format on
    }
    cout << "-----------------------------------------------------------------"
            "------------------\n\n";
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

----------------------------------------------
Element ID | Name | Atomic number | Mass (AMU)
---------- | ---- | ------------- | ----------
)gfm";

    for (const auto& el_key : element_map)
    {
        const auto& element = geometry.get_element(el_key.first);
        // clang-format off
        cout << setw(10) << std::left << el_key.first << " | "
             << setw(4) << element.name << " | "
             << setw(13) << element.atomic_number << " | "
             << element.atomic_mass << '\n';
        // clang-format on
    }
    cout << "----------------------------------------------" << endl;

    //// PRINT MATERIAL LIST ///

    const auto& material_map = geometry.matid_to_material_map();

    CELER_LOG(info) << "Loaded " << material_map.size() << " materials";
    cout << R"gfm(
## Materials

--------------------------------------------------------------------------------------------------
Material ID | Name                            | Composition
----------- | ------------------------------- | --------------------------------------------------
)gfm";

    for (const auto& mat_key : material_map)
    {
        const auto& material = geometry.get_material(mat_key.first);
        // clang-format off
        cout << setw(11) << std::left << mat_key.first << " | "
             << setw(31) << material.name << " |";
        // clang-format on
        for (const auto& key : material.elements_fractions)
        {
            cout << " " << geometry.get_element(key.first).name;
        }
        cout << '\n';
    }
    cout << "-----------------------------------------------------------------"
            "---------------------------------"
         << endl;

    //// PRINT VOLUME AND MATERIAL LIST ////

    const auto& volume_material_map = geometry.volid_to_matid_map();

    CELER_LOG(info) << "Loaded " << volume_material_map.size() << " volumes";
    cout << R"gfm(
## Volumes

--------------------------------------------------------------------------------------------
Volume ID | Material ID | Solid Name                           | Material Name
--------- | ----------- | ------------------------------------ | ---------------------------
)gfm";

    for (const auto& key_value : volume_material_map)
    {
        auto volid    = key_value.first;
        auto matid    = key_value.second;
        auto volume   = geometry.get_volume(volid);
        auto material = geometry.get_material(matid);

        // clang-format off
        cout << setw(9) << std::left << volid << " | "
             << setw(11) << matid << " | "
             << setw(36) << volume.solid_name << " | "
             << material.name << '\n';
        // clang-format on
    }
    cout << "-----------------------------------------------------------------"
            "---------------------------"
         << endl;
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
        CELER_LOG(critical) << "Exception while reading ROOT file'" << argv[1]
                            << "': " << e.what();
        return EXIT_FAILURE;
    }

    CELER_LOG(info) << "Successfully loaded ROOT file'" << argv[1] << "'";

    print_particles(*data.particle_params);

    CELER_LOG(info) << "Loaded " << data.physics_tables->size()
                    << " physics tables";
    cout << "# Physics tables\n\n";
    for (const ImportPhysicsTable& table : *data.physics_tables)
    {
        print_physics_table(table);
    }
    print_geometry(*data.geometry);

    return EXIT_SUCCESS;
}
