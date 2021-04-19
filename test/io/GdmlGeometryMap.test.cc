//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GdmlGeometryMap.test.cc
//---------------------------------------------------------------------------//
#include "io/GdmlGeometryMap.hh"
#include "io/RootImporter.hh"
#include "io/ImportData.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GdmlGeometryMapTest : public celeritas::Test
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

TEST_F(GdmlGeometryMapTest, import_geometry)
{
    GdmlGeometryMap geometry(data_);
    const auto      map = geometry.volid_to_matid_map();
    EXPECT_EQ(map.size(), 5);

    // Fetch a given ImportVolume provided a vol_id
    vol_id       volid  = 0;
    ImportVolume volume = geometry.get_volume(volid);
    EXPECT_EQ(volume.name, "box");

    // Fetch respective mat_id and ImportMaterial from the given vol_id
    mat_id         matid    = geometry.get_matid(volid);
    ImportMaterial material = geometry.get_material(matid);

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

    // Test elements within material;
    std::vector<std::string> names;
    std::vector<int>         atomic_numbers;
    std::vector<double>      number_fractions;
    std::vector<double>      atomic_masses;

    for (auto& elem_comp : material.elements)
    {
        auto element = geometry.get_element(elem_comp.element_id);
        names.push_back(element.name);
        atomic_numbers.push_back(element.atomic_number);
        number_fractions.push_back(elem_comp.number_fraction);
        atomic_masses.push_back(element.atomic_mass);
    }

    const std::string expected_names[]            = {"Fe", "Cr", "Ni"};
    const int         expected_atomic_numbers[]   = {26, 24, 28};
    const double      expected_number_fractions[] = {0.74, 0.18, 0.08};
    const double      expected_atomic_masses[]
        = {55.845110798, 51.996130137, 58.6933251009};

    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_atomic_numbers, atomic_numbers);
    EXPECT_VEC_SOFT_EQ(expected_number_fractions, number_fractions);
    EXPECT_VEC_SOFT_EQ(expected_atomic_masses, atomic_masses);
}
