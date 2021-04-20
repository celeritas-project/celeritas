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
TEST_F(RootImporterTest, elements)
{
    const auto elements = data_.elements;
    EXPECT_EQ(4, elements.size());

    std::vector<std::string>  names;
    std::vector<unsigned int> ids;
    std::vector<int>          atomic_numbers;
    std::vector<double>       atomic_masses, rad_lenghts_tsai, coulomb_factors;

    for (const auto& element : elements)
    {
        names.push_back(element.name);
        ids.push_back(element.element_id);
        atomic_masses.push_back(element.atomic_mass);
        atomic_numbers.push_back(element.atomic_number);
        coulomb_factors.push_back(element.coulomb_factor);
        rad_lenghts_tsai.push_back(element.radiation_length_tsai);
    }

    // clang-format off
    const std::string  expected_names[]          = {"Fe", "Cr", "Ni", "H"};
    const unsigned int expected_ids[]            = {0, 1, 2, 3};
    const int          expected_atomic_numbers[] = {26, 24, 28, 1};
    const double       expected_atomic_masses[]  = {55.845110798, 51.996130137,
        58.6933251009, 1.007940752665}; // [AMU]
    const double expected_coulomb_factors[] = {0.04197339849163,
        0.03592322294658, 0.04844802666907, 6.400838295295e-05};
    const double expected_rad_lenghts_tsai[] = {1.073632177106e-41,
        9.256561295161e-42, 1.231638535009e-41, 4.253575226044e-44};
    // clang-format on

    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_ids, ids);
    EXPECT_VEC_EQ(expected_atomic_numbers, atomic_numbers);
    EXPECT_VEC_SOFT_EQ(expected_atomic_masses, atomic_masses);
    EXPECT_VEC_SOFT_EQ(expected_coulomb_factors, coulomb_factors);
    EXPECT_VEC_SOFT_EQ(expected_rad_lenghts_tsai, rad_lenghts_tsai);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, materials)
{
    const auto materials = data_.materials;
    EXPECT_EQ(2, materials.size());

    std::vector<std::string>  names;
    std::vector<unsigned int> material_ids;
    std::vector<int>          states;
    std::vector<int>          pdgs;
    std::vector<double>       cutoff_energies, cutoff_ranges;
    std::vector<double> el_comps_ids, el_comps_mass_frac, el_comps_num_fracs;
    std::vector<double> densities, num_densities, e_densities, temperatures,
        rad_lengths, nuc_int_lenghts;

    for (const auto material : materials)
    {
        names.push_back(material.name);
        material_ids.push_back(material.material_id);
        states.push_back((int)material.state);
        densities.push_back(material.density);
        e_densities.push_back(material.electron_density);
        num_densities.push_back(material.number_density);
        nuc_int_lenghts.push_back(material.nuclear_int_length);
        rad_lengths.push_back(material.radiation_length);
        temperatures.push_back(material.temperature);

        for (const auto key : material.pdg_cutoffs)
        {
            pdgs.push_back(key.first);
            cutoff_energies.push_back(key.second.energy);
            cutoff_ranges.push_back(key.second.range);
        }

        for (const auto el_comp : material.elements)
        {
            el_comps_ids.push_back(el_comp.element_id);
            el_comps_mass_frac.push_back(el_comp.mass_fraction);
            el_comps_num_fracs.push_back(el_comp.number_fraction);
        }
    }

    // clang-format off
    const std::string expected_names[] = {"G4_Galactic", "G4_STAINLESS-STEEL"};
    const unsigned int expected_material_ids[] = {0, 1};
    const int          expected_states[]       = {3, 1};
    const int    expected_pdgs[] = {-11, 11, 22, 2212, -11, 11, 22, 2212};
    const double expected_cutoff_energies[] = {0.00099, 0.00099, 0.00099, 0.07,
        0.9260901525621, 0.9706947116044, 0.01733444524846, 0.07};
    const double expected_cutoff_ranges[]
        = {0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07};
    const double expected_densities[] = {1e-25, 8};
    const double expected_e_densities[]
        = {0.05974697167543, 2.244432022882e+24};
    const double expected_num_densities[]
        = {0.05974697167543, 8.699348925899e+22};
    const double expected_nuc_int_lenghts[]
        = {3.500000280825e+26, 16.67805709739};
    const double expected_rad_lengths[] = {6.304350904227e+26, 1.738067064483};
    const double expected_temperatures[] = {2.73, 293.15};
    const double expected_el_comps_ids[] = {3, 0, 1, 2};
    const double expected_el_comps_mass_frac[]
        = {1, 0.7462128746215, 0.1690010443115, 0.08478608106695};
    const double expected_el_comps_num_fracs[] = {1, 0.74, 0.18, 0.08};
    // clang-format on

    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_material_ids, material_ids);
    EXPECT_VEC_EQ(expected_states, states);
    EXPECT_VEC_EQ(expected_pdgs, pdgs);
    EXPECT_VEC_SOFT_EQ(expected_cutoff_energies, cutoff_energies);
    EXPECT_VEC_SOFT_EQ(expected_cutoff_ranges, cutoff_ranges);
    EXPECT_VEC_SOFT_EQ(expected_densities, densities);
    EXPECT_VEC_SOFT_EQ(expected_e_densities, e_densities);
    EXPECT_VEC_SOFT_EQ(expected_num_densities, num_densities);
    EXPECT_VEC_SOFT_EQ(expected_nuc_int_lenghts, nuc_int_lenghts);
    EXPECT_VEC_SOFT_EQ(expected_rad_lengths, rad_lengths);
    EXPECT_VEC_SOFT_EQ(expected_temperatures, temperatures);
    EXPECT_VEC_SOFT_EQ(expected_el_comps_ids, el_comps_ids);
    EXPECT_VEC_SOFT_EQ(expected_el_comps_mass_frac, el_comps_mass_frac);
    EXPECT_VEC_SOFT_EQ(expected_el_comps_num_fracs, el_comps_num_fracs);
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
TEST_F(RootImporterTest, volumes)
{
    const auto volumes = data_.volumes;
    EXPECT_EQ(5, volumes.size());

    std::vector<unsigned int> volume_ids;
    std::vector<unsigned int> material_ids;
    std::vector<std::string>  names, solids;

    for (const auto& volume : volumes)
    {
        volume_ids.push_back(volume.volume_id);
        material_ids.push_back(volume.material_id);
        names.push_back(volume.name);
        solids.push_back(volume.solid_name);
    }

    const unsigned int expected_volume_ids[]   = {4, 0, 1, 2, 3};
    const unsigned int expected_material_ids[] = {0, 1, 1, 1, 1};
    const std::string  expected_names[]
        = {"World", "box", "boxReplica", "boxReplica", "boxReplica"};
    const std::string expected_solids[]
        = {"World", "box", "boxReplica", "boxReplica2", "boxReplica3"};

    EXPECT_VEC_EQ(expected_volume_ids, volume_ids);
    EXPECT_VEC_EQ(expected_material_ids, material_ids);
    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_solids, solids);
}
