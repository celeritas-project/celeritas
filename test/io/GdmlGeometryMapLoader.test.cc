//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GdmlGeometryMapLoader.test.cc
//---------------------------------------------------------------------------//
#include "io/GdmlGeometryMapLoader.hh"

#include "io/RootLoader.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GdmlGeometryMapLoaderTest : public celeritas::Test
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

TEST_F(GdmlGeometryMapLoaderTest, load_gdml_geometry)
{
    RootLoader            root_loader(this->root_filename_.c_str());
    GdmlGeometryMapLoader geometry_loader(root_loader);
    const auto            geometry = geometry_loader();

    auto map = geometry->volid_to_matid_map();
    EXPECT_EQ(map.size(), 5);

    // Fetch a given ImportVolume provided a vol_id
    vol_id       volid  = 0;
    ImportVolume volume = geometry->get_volume(volid);
    EXPECT_EQ(volume.name, "box");

    // Fetch respective mat_id and ImportMaterial from the given vol_id
    mat_id         matid    = geometry->get_matid(volid);
    ImportMaterial material = geometry->get_material(matid);

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
        auto element = geometry->get_element(elid);

        EXPECT_EQ(elements_name[i], element.name);
        EXPECT_EQ(atomic_number[i], element.atomic_number);
        EXPECT_SOFT_EQ(atomic_mass[i], element.atomic_mass);
        EXPECT_SOFT_EQ(fraction[i], iter.second);
        i++;
    }
}
