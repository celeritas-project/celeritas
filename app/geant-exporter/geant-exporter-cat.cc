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
 * Dump the contents of a ROOT file writen by geant-exporter.
 */
int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        // If number of arguments is incorrect, print help
        cout << "Usage: " << argv[0] << " output.root" << endl;
        return 1;
    }

    std::shared_ptr<const ParticleParams> particles;
    std::shared_ptr<GdmlGeometryMap>      geometry;

    try
    {
        RootImporter import(argv[1]);

        auto data = import();
        particles = data.particle_params;
        geometry  = data.geometry;
    }
    catch (const DebugError& e)
    {
        cout << "Exception while reading ROOT file'" << argv[1] << "':\n"
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    // /// PRINT PARTICLE LIST /// //

    const auto& all_md = particles->md();

    cout << endl;
    cout << ">>> Loaded " << all_md.size() << " particles from `" << argv[1]
         << "`.\n";

    cout << R"gfm(
-----------------------------------------------------------------------
Name              | PDG Code    | Mass [MeV] | Charge [e] | Decay [1/s]
----------------- | ----------- | ---------- | ---------- | -----------
)gfm";

    ParticleDefId::value_type def_id = 0;
    for (const auto& md : all_md)
    {
        const ParticleDef& def = particles->get(ParticleDefId{def_id++});

        // clang-format off
        cout << setw(17) << std::left << md.name << " | "
             << setw(11) << md.pdg_code.get() << " | "
             << setw(10) << setprecision(6) << def.mass.value() << " | "
             << setw(10) << setprecision(3) << def.charge.value() << " | "
             << setw(11) << setprecision(3) << def.decay_constant
             << '\n';
        // clang-format on
    }
    cout << "-----------------------------------------------------------------"
            "------"
         << endl;

    // /// PRINT ELEMENT LIST /// //

    const auto& element_map = geometry->elemid_to_element_map();

    cout << endl;
    cout << ">>> Loaded " << element_map.size() << " elements from `"
         << argv[1] << "`." << std::endl;
    cout << R"gfm(
----------------------------------------------
Element ID | Name | Atomic number | Mass (AMU)
---------- | ---- | ------------- | ----------
)gfm";

    for (const auto& el_key : element_map)
    {
        const auto& element = geometry->get_element(el_key.first);
        // clang-format off
        cout << setw(10) << std::left << el_key.first << " | "
             << setw(4) << element.name << " | "
             << setw(13) << element.atomic_number << " | "
             << element.atomic_mass << '\n';
        // clang-format on
    }
    cout << "----------------------------------------------" << endl;

    // /// PRINT MATERIAL LIST /// //

    const auto& material_map = geometry->matid_to_material_map();

    cout << endl;
    cout << ">>> Loaded " << material_map.size() << " materials from `"
         << argv[1] << "`." << std::endl;
    cout << R"gfm(
--------------------------------------------------------------------------------------------------
Material ID | Name                            | Composition
----------- | ------------------------------- | --------------------------------------------------
)gfm";

    for (const auto& mat_key : material_map)
    {
        const auto& material = geometry->get_material(mat_key.first);
        // clang-format off
        cout << setw(11) << std::left << mat_key.first << " | "
             << setw(31) << material.name << " | ";
        // clang-format on
        for (const auto& key : material.elements_fractions)
        {
            cout << geometry->get_element(key.first).name << " ";
        }
        cout << endl;
    }
    cout << "-----------------------------------------------------------------"
            "---------------------------------"
         << endl;

    // /// PRINT VOLUME AND MATERIAL LIST /// //

    const auto& volume_material_map = geometry->volid_to_matid_map();

    cout << endl;
    cout << ">>> Loaded " << volume_material_map.size() << " volumes from `"
         << argv[1] << "`." << std::endl;
    cout << R"gfm(
--------------------------------------------------------------------------------------------
Volume ID | Material ID | Solid Name                           | Material Name
--------- | ----------- | ------------------------------------ | ---------------------------
)gfm";

    for (const auto& key_value : volume_material_map)
    {
        auto volid    = key_value.first;
        auto matid    = key_value.second;
        auto volume   = geometry->get_volume(volid);
        auto material = geometry->get_material(matid);

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
    cout << endl;

    return EXIT_SUCCESS;
}
