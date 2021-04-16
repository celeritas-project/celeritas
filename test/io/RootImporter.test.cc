//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootImporter.test.cc
//---------------------------------------------------------------------------//
#include "io/RootImporter.hh"

#include <algorithm>
#include "io/ImportPhysicsTable.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/base/CutoffView.hh"
#include "physics/material/MaterialView.hh"
#include "base/Types.hh"
#include "base/Range.hh"
#include "io/ImportData.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
/*!
 * The \e geant-exporter-data.root is created by the \e app/geant-exporter
 * using the \e four-steel-slabs.gdml example file available in
 * \e app/geant-exporter/data.
 *
 * \note
 * G4EMLOW7.12 and G4EMLOW7.13 produce slightly different physics vector
 * values for steel, failing \c processes test.
 */

class RootImporterTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        root_filename_ = this->test_data_path("io", "geant-exporter-data.root");

        RootImporter import_from_root(root_filename_.c_str());
        data_ = import_from_root();
    }

    std::string root_filename_;
    ImportData  data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, particles)
{
    const auto particles = data_.particles;
    EXPECT_EQ(19, particles.size());

    // Check all names/PDG codes
    std::vector<std::string> loaded_names;
    std::vector<int>         loaded_pdgs;
    for (const auto& particle : particles)
    {
        loaded_names.push_back(particle.name);
        loaded_pdgs.push_back(particle.pdg);
    }

    // Particle ordering is the same as in the ROOT file
    // clang-format off
    const std::string expected_loaded_names[] = {"He3", "alpha", "anti_He3",
    "anti_alpha", "anti_deuteron", "anti_proton", "anti_triton", "deuteron",
    "e+", "e-", "gamma", "kaon+", "kaon-", "mu+", "mu-", "pi+", "pi-",
    "proton", "triton"};

    const int expected_loaded_pdgs[] = {1000020030, 1000020040, -1000020030,
    -1000020040, -1000010020, -2212, -1000010030, 1000010020, -11, 11, 22, 321,
    -321, -13, 13, 211, -211, 2212, 1000010030};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, processes)
{
    const auto processes = data_.processes;

    auto iter = std::find_if(
        processes.begin(), processes.end(), [](const ImportProcess& proc) {
            return PDGNumber{proc.particle_pdg} == celeritas::pdg::electron()
                   && proc.process_class == ImportProcessClass::e_ioni;
        });
    ASSERT_NE(processes.end(), iter);

    EXPECT_EQ(ImportProcessType::electromagnetic, iter->process_type);
    ASSERT_EQ(1, iter->models.size());
    EXPECT_EQ(ImportModelClass::moller_bhabha, iter->models.front());

    const auto& tables = iter->tables;
    ASSERT_EQ(3, tables.size());
    {
        // Test energy loss table
        const ImportPhysicsTable& dedx = tables[0];
        ASSERT_EQ(ImportTableType::dedx, dedx.table_type);
        EXPECT_EQ(ImportUnits::mev, dedx.x_units);
        EXPECT_EQ(ImportUnits::mev_per_cm, dedx.y_units);
        ASSERT_EQ(2, dedx.physics_vectors.size());

        const ImportPhysicsVector& steel = dedx.physics_vectors.back();
        EXPECT_EQ(ImportPhysicsVectorType::log, steel.vector_type);
        ASSERT_EQ(steel.x.size(), steel.y.size());
        ASSERT_EQ(85, steel.x.size());
        EXPECT_SOFT_EQ(1e-4, steel.x.front());
        EXPECT_SOFT_EQ(1e8, steel.x.back());
        EXPECT_SOFT_EQ(839.66835335480653, steel.y.front());
        EXPECT_SOFT_EQ(11.207442027393293, steel.y.back());
    }
    {
        // Test range table
        const ImportPhysicsTable& range = tables[1];
        ASSERT_EQ(ImportTableType::range, range.table_type);
        EXPECT_EQ(ImportUnits::mev, range.x_units);
        EXPECT_EQ(ImportUnits::cm, range.y_units);
        ASSERT_EQ(2, range.physics_vectors.size());

        const ImportPhysicsVector& steel = range.physics_vectors.back();
        EXPECT_EQ(ImportPhysicsVectorType::log, steel.vector_type);
        ASSERT_EQ(steel.x.size(), steel.y.size());
        ASSERT_EQ(85, steel.x.size());
        EXPECT_SOFT_EQ(1e-4, steel.x.front());
        EXPECT_SOFT_EQ(1e8, steel.x.back());
        EXPECT_SOFT_EQ(2.3818927937550707e-07, steel.y.front());
        EXPECT_SOFT_EQ(8922642.7361662444, steel.y.back());
    }
    {
        // Test cross section table
        const ImportPhysicsTable& xs = tables[2];
        ASSERT_EQ(ImportTableType::lambda, xs.table_type);
        EXPECT_EQ(ImportUnits::mev, xs.x_units);
        EXPECT_EQ(ImportUnits::cm_inv, xs.y_units);
        ASSERT_EQ(2, xs.physics_vectors.size());

        const ImportPhysicsVector& steel = xs.physics_vectors.back();
        EXPECT_EQ(ImportPhysicsVectorType::log, steel.vector_type);
        ASSERT_EQ(steel.x.size(), steel.y.size());
        ASSERT_EQ(55, steel.x.size());
        EXPECT_SOFT_EQ(1.9413894232088691, steel.x.front());
        EXPECT_SOFT_EQ(1e8, steel.x.back());
        EXPECT_SOFT_EQ(0, steel.y.front());
        EXPECT_SOFT_EQ(0.24960554333948043, steel.y[1]);
        EXPECT_SOFT_EQ(0.58950471707787622, steel.y.back());
    }
}

//---------------------------------------------------------------------------//
#if 0
TEST_F(RootImporterTest, geometry)
{
    const auto map = data_.geometry.volid_to_matid_map();
    EXPECT_EQ(5, map.size());

    // Fetch a given ImportVolume provided a vol_id
    vol_id       volid  = 0;
    ImportVolume volume = data_.geometry.get_volume(volid);
    EXPECT_EQ(volume.name, "box");

    // Fetch respective mat_id and ImportMaterial from the given vol_id
    mat_id         matid    = data_.geometry.get_matid(volid);
    ImportMaterial material = data_.geometry.get_material(matid);

    // Test material
    EXPECT_EQ(1, matid);
    EXPECT_EQ("G4_STAINLESS-STEEL", material.name);
    EXPECT_EQ(ImportMaterialState::solid, material.state);
    EXPECT_SOFT_EQ(293.15, material.temperature); // [K]
    EXPECT_SOFT_EQ(8, material.density);          // [g/cm^3]
    EXPECT_SOFT_EQ(2.2444320228819809e+24,
                   material.electron_density); // [1/cm^3]
    EXPECT_SOFT_EQ(8.6993489258991514e+22, material.number_density); // [1/cm^3]
    EXPECT_SOFT_EQ(1.738067064482842, material.radiation_length);    // [cm]
    EXPECT_SOFT_EQ(16.678057097389537, material.nuclear_int_length); // [cm]
    EXPECT_EQ(3, material.elements.size());

    // Test elements within material
    static const int array_size                = 3;
    std::string      elements_name[array_size] = {"Fe", "Cr", "Ni"};
    unsigned int     atomic_number[array_size] = {26, 24, 28};
    real_type        fraction[array_size]
        = {0.74621287462152097, 0.16900104431152499, 0.0847860810669534};
    real_type atomic_mass[array_size]
        = {55.845110798, 51.996130136999994, 58.693325100900005}; // [AMU]

    int i = 0;
    for (auto& elem_comp : material.elements)
    {
        auto element = data_.geometry.get_element(elem_comp.element_id);

        EXPECT_EQ(elements_name[i], element.name);
        EXPECT_EQ(atomic_number[i], element.atomic_number);
        EXPECT_SOFT_EQ(atomic_mass[i], element.atomic_mass);
        EXPECT_SOFT_EQ(fraction[i], elem_comp.mass_fraction);
        i++;
    }
}
#endif

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, material_cutoffs)
{
    const auto materials = data_.materials;
    EXPECT_EQ(2, materials.size());

    std::vector<int>    pdgs;
    std::vector<double> energies, ranges;

    for (const auto material : materials)
    {
        for (const auto key : material.pdg_cutoffs)
        {
            pdgs.push_back(key.first);
            energies.push_back(key.second.energy);
            ranges.push_back(key.second.range);
        }
    }

    // clang-format off
    const int expected_pdgs[] = {-11, 11, 22, 2212, -11, 11, 22, 2212};

    const double expected_energies[] = {0.00099, 0.00099, 0.00099, 0.07,
    0.9260901525621, 0.9706947116044, 0.01733444524846, 0.07};

    const double expected_ranges[] = {0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
    0.07};
    // clang-format on

    EXPECT_VEC_EQ(expected_pdgs, pdgs);
    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}
