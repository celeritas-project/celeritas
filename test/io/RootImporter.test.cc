//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootImporter.test.cc
//---------------------------------------------------------------------------//

#include "io/RootImporter.hh"
#include "io/ImportPhysicsTable.hh"
#include "physics/base/ParticleMd.hh"
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
    RootImporter import(root_filename_.c_str());
    auto         data = import();

    EXPECT_EQ(19, data.particle_params->size());

    EXPECT_GE(data.particle_params->find(PDGNumber(11)).get(), 0);
    ParticleDefId electron_id = data.particle_params->find(PDGNumber(11));
    ParticleDef   electron    = data.particle_params->get(electron_id);

    EXPECT_SOFT_EQ(0.510998910, electron.mass.value());
    EXPECT_EQ(-1, electron.charge.value());
    EXPECT_EQ(0, electron.decay_constant);

    std::vector<std::string> loaded_names;
    std::vector<int>         loaded_pdgs;
    for (const auto& md : data.particle_params->md())
    {
        loaded_names.push_back(md.name);
        loaded_pdgs.push_back(md.pdg_code.get());
    }

    // clang-format off
    const std::string expected_loaded_names[] = {"gamma", "e+", "e-", "mu+",
        "mu-", "pi-", "pi+", "kaon-", "kaon+", "anti_proton", "proton",
        "anti_deuteron", "deuteron", "anti_He3", "He3", "anti_triton",
        "triton", "anti_alpha", "alpha"};
    const int expected_loaded_pdgs[] = {22, -11, 11, -13, 13, -211, 211, -321,
        321, -2212, 2212, -1000010020, 1000010020, -1000020030, 1000020030,
        -1000010030, 1000010030, -1000020040, 1000020040};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_tables)
{
    RootImporter import(root_filename_.c_str());
    auto         data = import();

    EXPECT_GE(data.physics_tables->size(), 0);

    // Test table search
    bool lambda_kn_gamma_table = false;
    for (auto table : *data.physics_tables)
    {
        EXPECT_GE(table.physics_vectors.size(), 0);

        if (table.particle == PDGNumber{celeritas::pdg::gamma()}
            && table.table_type == ImportTableType::lambda
            && table.process == ImportProcess::compton
            && table.model == ImportModel::klein_nishina)
        {
            lambda_kn_gamma_table = true;
            break;
        }
    }
    EXPECT_TRUE(lambda_kn_gamma_table);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_geometry)
{
    RootImporter import(root_filename_.c_str());
    auto         data = import();

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
    RootImporter import(root_filename_.c_str());
    auto         data = import();

    // Material labels
    std::string material_label;
    material_label = data.material_params->id_to_label(MaterialDefId{0});
    EXPECT_EQ(material_label, "G4_Galactic");
    material_label = data.material_params->id_to_label(MaterialDefId{1});
    EXPECT_EQ(material_label, "G4_STAINLESS-STEEL");

    auto mat_host_ptr = data.material_params->host_pointers();

    /*!
     * Material
     *
     * Geant4 has outdated constants. The discrepancy between Geant4 /
     * Celeritas constants results in the slightly different numerical values
     * calculated by Celeritas.
     */
    auto material = mat_host_ptr.materials[1];

    EXPECT_EQ(MatterState::solid, material.matter_state);
    EXPECT_SOFT_EQ(293.15, material.temperature);         // [K]
    EXPECT_SOFT_EQ(8.0000013655195588, material.density); // [g/cm^3]
    EXPECT_SOFT_EQ(2.2444324067595884e+24,
                   material.electron_density); // [1/cm^3]
    EXPECT_SOFT_EQ(8.6993504137968536e+22, material.number_density); // [1/cm^3]
    EXPECT_EQ(3, material.elements.size());

    // Elements of a material
    // Fractions are normalized and thus may differ from the imported ones
    const int array_size = 3;
    // Fe, Cr, Ni
    ElementDefId element_def_id[array_size]
        = {ElementDefId{0}, ElementDefId{1}, ElementDefId{2}};
    real_type fraction[array_size] = {0.74, 0.18, 0.08};

    for (auto i : celeritas::range(material.elements.size()))
    {
        EXPECT_EQ(material.elements[i].element, element_def_id[i]);
        EXPECT_SOFT_EQ(material.elements[i].fraction, fraction[i]);
    }
}
