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
#include "physics/material/MaterialView.hh"
#include "base/Types.hh"
#include "base/Range.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
/*!
 * The geant-exporter-data.root is created by the app/geant-exporter using the
 * four-steel-slabs.gdml example file available in app/geant-exporter/data
 */
class RootImporterTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        root_filename_ = this->test_data_path("io", "geant-exporter-data.root");
    }

    std::string root_filename_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_particles)
{
    RootImporter import_from_root(root_filename_.c_str());
    auto         data = import_from_root();

    const auto& particles = *data.particle_params;

    EXPECT_EQ(19, particles.size());

    // Check electron data
    ParticleId electron_id = data.particle_params->find(PDGNumber(11));
    ASSERT_GE(electron_id.get(), 0);
    ParticleDef electron = data.particle_params->get(electron_id);
    EXPECT_SOFT_EQ(0.510998910, electron.mass.value());
    EXPECT_EQ(-1, electron.charge.value());
    EXPECT_EQ(0, electron.decay_constant);

    // Check all names/PDG codes
    std::vector<std::string> loaded_names;
    std::vector<int>         loaded_pdgs;
    for (auto idx : range<ParticleId::value_type>(particles.size()))
    {
        ParticleId particle_id{idx};
        loaded_names.push_back(particles.id_to_label(particle_id));
        loaded_pdgs.push_back(particles.id_to_pdg(particle_id).get());
    }

    // clang-format off
    const std::string expected_loaded_names[] = {"gamma", "e-", "e+", "mu-",
        "mu+", "pi+", "pi-", "kaon+", "kaon-", "proton", "anti_proton",
        "deuteron", "anti_deuteron", "He3", "anti_He3", "triton",
        "anti_triton", "alpha", "anti_alpha"};
    const int expected_loaded_pdgs[] = {22, 11, -11, 13, -13, 211, -211, 321,
        -321, 2212, -2212, 1000010020, -1000010020, 1000020030, -1000020030,
        1000010030, -1000010030, 1000020040, -1000020040};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_processes)
{
    RootImporter import_from_root(root_filename_.c_str());
    auto         processes = import_from_root().processes;

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
        EXPECT_SOFT_EQ(839.66834289225289, steel.y.front());
        EXPECT_SOFT_EQ(11.207441942857839, steel.y.back());
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
        EXPECT_SOFT_EQ(2.3818928234342666e-07, steel.y.front());
        EXPECT_SOFT_EQ(8922642.803467935, steel.y.back());
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
        EXPECT_SOFT_EQ(0.24960554022818102, steel.y[1]);
        EXPECT_SOFT_EQ(0.58950470972977953, steel.y.back());
    }
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_geometry)
{
    RootImporter import_from_root(root_filename_.c_str());
    auto         data = import_from_root();

    auto map = data.geometry->volid_to_matid_map();
    EXPECT_EQ(map.size(), 5);

    // Fetch a given ImportVolume provided a vol_id
    vol_id       volid  = 0;
    ImportVolume volume = data.geometry->get_volume(volid);
    EXPECT_EQ(volume.name, "box");

    // Fetch respective mat_id and ImportMaterial from the given vol_id
    mat_id         matid    = data.geometry->get_matid(volid);
    ImportMaterial material = data.geometry->get_material(matid);

    // Test material
    EXPECT_EQ(1, matid);
    EXPECT_EQ("G4_STAINLESS-STEEL", material.name);
    EXPECT_EQ(ImportMaterialState::solid, material.state);
    EXPECT_SOFT_EQ(293.15, material.temperature); // [K]
    EXPECT_SOFT_EQ(8, material.density);          // [g/cm^3]
    EXPECT_SOFT_EQ(2.2444324067595881e+24,
                   material.electron_density); // [1/cm^3]
    EXPECT_SOFT_EQ(8.6993504137968536e+22, material.number_density); // [1/cm^3]
    EXPECT_SOFT_EQ(1.7380670928095856, material.radiation_length);   // [cm]
    EXPECT_SOFT_EQ(16.678055775064472, material.nuclear_int_length); // [cm]
    EXPECT_EQ(3, material.elements_fractions.size());

    // Test elements within material
    static const int array_size                = 3;
    std::string      elements_name[array_size] = {"Fe", "Cr", "Ni"};
    int              atomic_number[array_size] = {26, 24, 28};
    real_type        fraction[array_size]
        = {0.74621287462152097, 0.16900104431152499, 0.0847860810669534};
    real_type atomic_mass[array_size]
        = {55.845110798, 51.996130136999994, 58.693325100900005}; // [AMU]

    int i = 0;
    for (auto const& iter : material.elements_fractions)
    {
        auto elid    = iter.first;
        auto element = data.geometry->get_element(elid);

        EXPECT_EQ(elements_name[i], element.name);
        EXPECT_EQ(atomic_number[i], element.atomic_number);
        EXPECT_SOFT_EQ(atomic_mass[i], element.atomic_mass);
        EXPECT_SOFT_EQ(fraction[i], iter.second);
        i++;
    }
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_material_params)
{
    RootImporter import_from_root(root_filename_.c_str());
    auto         data = import_from_root();

    // Material labels
    std::string material_label;
    material_label = data.material_params->id_to_label(MaterialId{0});
    EXPECT_EQ(material_label, "G4_Galactic");
    material_label = data.material_params->id_to_label(MaterialId{1});
    EXPECT_EQ(material_label, "G4_STAINLESS-STEEL");

    /*!
     * Material
     *
     * Geant4 has outdated constants. The discrepancy between Geant4 /
     * Celeritas constants results in the slightly different numerical values
     * calculated by Celeritas.
     */
    celeritas::MaterialView mat(data.material_params->host_pointers(),
                                MaterialId{1});

    EXPECT_EQ(MatterState::solid, mat.matter_state());
    EXPECT_SOFT_EQ(293.15, mat.temperature());         // [K]
    EXPECT_SOFT_EQ(8.0000013655195588, mat.density()); // [g/cm^3]
    EXPECT_SOFT_EQ(2.2444324067595884e+24,
                   mat.electron_density());                       // [1/cm^3]
    EXPECT_SOFT_EQ(8.6993504137968536e+22, mat.number_density()); // [1/cm^3]

    // Test elements by unpacking them
    std::vector<unsigned int> els;
    std::vector<real_type>    fracs;
    for (const auto& component : mat.elements())
    {
        els.push_back(component.element.unchecked_get());
        fracs.push_back(component.fraction);
    }

    // Fractions are normalized and thus may differ from the imported ones
    // Fe, Cr, Ni
    static unsigned int const expected_els[]   = {0, 1, 2};
    static real_type          expected_fracs[] = {0.74, 0.18, 0.08};
    EXPECT_VEC_EQ(expected_els, els);
    EXPECT_VEC_SOFT_EQ(expected_fracs, fracs);
}
