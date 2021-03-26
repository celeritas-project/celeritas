//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialParamsLoader.test.cc
//---------------------------------------------------------------------------//
#include "io/MaterialParamsLoader.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MaterialParamsLoaderTest : public celeritas::Test
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
TEST_F(MaterialParamsLoaderTest, load_material_params)
{
    RootLoader           root_loader(this->root_filename_.c_str());
    MaterialParamsLoader mat_params_loader(root_loader);

    const auto materials = mat_params_loader();

    // Material labels
    std::string material_label;
    material_label = materials->id_to_label(MaterialId{0});
    EXPECT_EQ(material_label, "G4_Galactic");
    material_label = materials->id_to_label(MaterialId{1});
    EXPECT_EQ(material_label, "G4_STAINLESS-STEEL");

    /*!
     * Material
     *
     * Geant4 has outdated constants. The discrepancy between Geant4 /
     * Celeritas constants results in the slightly different numerical values
     * calculated by Celeritas.
     */
    celeritas::MaterialView mat(materials->host_pointers(), MaterialId{1});

    EXPECT_EQ(MatterState::solid, mat.matter_state());
    EXPECT_SOFT_EQ(293.15, mat.temperature());         // [K]
    EXPECT_SOFT_EQ(7.9999999972353661, mat.density()); // [g/cm^3]
    EXPECT_SOFT_EQ(2.2444320228819809e+24,
                   mat.electron_density());                       // [1/cm^3]
    EXPECT_SOFT_EQ(8.6993489258991514e+22, mat.number_density()); // [1/cm^3]

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
