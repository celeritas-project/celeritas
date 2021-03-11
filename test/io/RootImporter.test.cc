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

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
/*!
 * The geant-exporter-data.root is created by the app/geant-exporter using the
 * four-steel-slabs.gdml example file available in app/geant-exporter/data.
 *
 * \note
 * G4EMLOW7.12 and G4EMLOW7.13 produce slightly different physics vector
 * values for steel, failing \c import_processes test.
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

    std::string               root_filename_;
    RootImporter::result_type data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_particles)
{
    const auto& particles = *data_.particle_params;

    EXPECT_EQ(19, particles.size());

    // Check electron data
    ParticleId electron_id = data_.particle_params->find(PDGNumber(11));
    ASSERT_GE(electron_id.get(), 0);
    const auto& electron = data_.particle_params->get(electron_id);
    EXPECT_SOFT_EQ(0.510998910, electron.mass().value());
    EXPECT_EQ(-1, electron.charge().value());
    EXPECT_EQ(0, electron.decay_constant());

    // Check all names/PDG codes
    std::vector<std::string> loaded_names;
    std::vector<int>         loaded_pdgs;
    for (auto particle_id : range(ParticleId{particles.size()}))
    {
        loaded_names.push_back(particles.id_to_label(particle_id));
        loaded_pdgs.push_back(particles.id_to_pdg(particle_id).get());
    }

    // Particle ordering is the same as in the ROOT file
    // clang-format off
    const std::string expected_loaded_names[] = {"He3", "alpha", "anti_He3", 
        "anti_alpha", "anti_deuteron", "anti_proton", "anti_triton", 
        "deuteron", "e+", "e-", "gamma", "kaon+", "kaon-", "mu+", "mu-", "pi+", 
        "pi-", "proton", "triton"};
    const int expected_loaded_pdgs[] = {1000020030, 1000020040, -1000020030, 
        -1000020040, -1000010020, -2212, -1000010030, 1000010020, -11, 11, 22, 
        321, -321, -13, 13, 211, -211, 2212, 1000010030};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_processes)
{
    const auto& processes = data_.processes;

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
        EXPECT_SOFT_EQ(11.205845009964834, steel.y.back());
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
        EXPECT_SOFT_EQ(8923914.3599599935, steel.y.back());
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
        EXPECT_SOFT_EQ(1.9359790960928149, steel.x.front());
        EXPECT_SOFT_EQ(1e8, steel.x.back());
        EXPECT_SOFT_EQ(0, steel.y.front());
        EXPECT_SOFT_EQ(0.24709010460842684, steel.y[1]);
        EXPECT_SOFT_EQ(0.59115215175950464, steel.y.back());
    }
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_geometry)
{
    auto map = data_.geometry->volid_to_matid_map();
    EXPECT_EQ(map.size(), 5);

    // Fetch a given ImportVolume provided a vol_id
    vol_id       volid  = 0;
    ImportVolume volume = data_.geometry->get_volume(volid);
    EXPECT_EQ(volume.name, "box");

    // Fetch respective mat_id and ImportMaterial from the given vol_id
    mat_id         matid    = data_.geometry->get_matid(volid);
    ImportMaterial material = data_.geometry->get_material(matid);

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
        auto element = data_.geometry->get_element(elid);

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
    // Material labels
    std::string material_label;
    material_label = data_.material_params->id_to_label(MaterialId{0});
    EXPECT_EQ(material_label, "G4_Galactic");
    material_label = data_.material_params->id_to_label(MaterialId{1});
    EXPECT_EQ(material_label, "G4_STAINLESS-STEEL");

    /*!
     * Material
     *
     * Geant4 has outdated constants. The discrepancy between Geant4 /
     * Celeritas constants results in the slightly different numerical values
     * calculated by Celeritas.
     */
    celeritas::MaterialView mat(data_.material_params->host_pointers(),
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

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_cutoffs)
{
    const auto& particles = *data_.particle_params;
    const auto& materials = *data_.material_params;
    const auto& cutoffs   = *data_.cutoff_params;

    std::vector<double> energies, ranges;

    for (auto i : range<ParticleId::size_type>(particles.size()))
    {
        for (auto j : range<MaterialId::size_type>(materials.size()))
        {
            CutoffView cutoff_view(
                cutoffs.host_pointers(), ParticleId{i}, MaterialId{j});

            energies.push_back(cutoff_view.energy().value());
            ranges.push_back(cutoff_view.range());
        }
    }

    // clang-format off
    const double expected_energies[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0.00099, 0.9174879161109, 0.00099, 0.9679895480464, 0.00099, 
        0.01728575113104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07, 0.07, 0, 0};

    const double expected_ranges[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0.07, 0.07, 0, 0};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}
