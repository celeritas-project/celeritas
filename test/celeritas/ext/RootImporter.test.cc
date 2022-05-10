//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/RootImporter.hh"

#include <algorithm>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
/*!
 * The \e four-steel-slabs.root is created by the \e app/geant-exporter
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
        root_filename_
            = this->test_data_path("celeritas", "four-steel-slabs.root");

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
    EXPECT_EQ(4, particles.size());

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
    const std::string expected_loaded_names[] = {"e+", "e-", "gamma", "proton"};

    const int expected_loaded_pdgs[] = {-11, 11, 22, 2212};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, elements)
{
    const auto elements = data_.elements;
    EXPECT_EQ(4, elements.size());

    std::vector<std::string> names;
    std::vector<int>         atomic_numbers;
    std::vector<double>      atomic_masses, rad_lenghts_tsai, coulomb_factors;

    for (const auto& element : elements)
    {
        names.push_back(element.name);
        atomic_masses.push_back(element.atomic_mass);
        atomic_numbers.push_back(element.atomic_number);
        coulomb_factors.push_back(element.coulomb_factor);
        rad_lenghts_tsai.push_back(element.radiation_length_tsai);
    }

    // clang-format off
    const std::string  expected_names[]          = {"Fe", "Cr", "Ni", "H"};
    const int          expected_atomic_numbers[] = {26, 24, 28, 1};
    const double       expected_atomic_masses[]  = {55.845110798, 51.996130137,
        58.6933251009, 1.007940752665}; // [AMU]
    const double expected_coulomb_factors[] = {0.04197339849163,
        0.03592322294658, 0.04844802666907, 6.400838295295e-05};
    const double expected_rad_lenghts_tsai[] = {1.073632177106e-41,
        9.256561295161e-42, 1.231638535009e-41, 4.253575226044e-44};
    // clang-format on

    EXPECT_VEC_EQ(expected_names, names);
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

    std::vector<std::string> names;
    std::vector<int>         states;
    std::vector<int>         pdgs;
    std::vector<double>      cutoff_energies, cutoff_ranges;
    std::vector<double> el_comps_ids, el_comps_mass_frac, el_comps_num_fracs;
    std::vector<double> densities, num_densities, e_densities, temperatures,
        rad_lengths, nuc_int_lenghts;

    for (const auto& material : materials)
    {
        names.push_back(material.name);
        states.push_back((int)material.state);
        densities.push_back(material.density);
        e_densities.push_back(material.electron_density);
        num_densities.push_back(material.number_density);
        nuc_int_lenghts.push_back(material.nuclear_int_length);
        rad_lengths.push_back(material.radiation_length);
        temperatures.push_back(material.temperature);

        for (const auto& key : material.pdg_cutoffs)
        {
            pdgs.push_back(key.first);
            cutoff_energies.push_back(key.second.energy);
            cutoff_ranges.push_back(key.second.range);
        }

        for (const auto& el_comp : material.elements)
        {
            el_comps_ids.push_back(el_comp.element_id);
            el_comps_mass_frac.push_back(el_comp.mass_fraction);
            el_comps_num_fracs.push_back(el_comp.number_fraction);
        }
    }

    // clang-format off
    const std::string expected_names[]  = {"G4_Galactic", "G4_STAINLESS-STEEL"};
    const int         expected_states[] = {3, 1};
    const int    expected_pdgs[] = {-11, 11, 22, 2212, -11, 11, 22, 2212};
    const double expected_cutoff_energies[] = {0.00099, 0.00099, 0.00099, 0.1,
        1.22808845964606, 1.31345289979559, 0.0209231725658313, 0.1};
    const double expected_cutoff_ranges[]
        = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
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
        EXPECT_SOFT_EQ(11.380485677836065, steel.y.back());
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
        EXPECT_SOFT_EQ(8786971.3079055995, steel.y.back());
    }
    {
        // Test cross-section table
        const ImportPhysicsTable& xs = tables[2];
        ASSERT_EQ(ImportTableType::lambda, xs.table_type);
        EXPECT_EQ(ImportUnits::mev, xs.x_units);
        EXPECT_EQ(ImportUnits::cm_inv, xs.y_units);
        ASSERT_EQ(2, xs.physics_vectors.size());

        const ImportPhysicsVector& steel = xs.physics_vectors.back();
        EXPECT_EQ(ImportPhysicsVectorType::log, steel.vector_type);
        ASSERT_EQ(steel.x.size(), steel.y.size());
        ASSERT_EQ(54, steel.x.size());
        EXPECT_SOFT_EQ(2.6269057995911775, steel.x.front());
        EXPECT_SOFT_EQ(1e8, steel.x.back());
        EXPECT_SOFT_EQ(0, steel.y.front());
        EXPECT_SOFT_EQ(0.18987862452122845, steel.y[1]);
        EXPECT_SOFT_EQ(0.43566778103861714, steel.y.back());
    }
    {
        // Test element selector cross-section table
        EXPECT_EQ(iter->models.size(), iter->micro_xs.size());

        // Current iter points to process class e_ioni
        // Process e_ioni currently has only one available model: Moller-Bhabha
        EXPECT_EQ(1, iter->micro_xs.size());

        const auto& mb_pair = iter->micro_xs.find(iter->models.front());
        EXPECT_EQ(ImportModelClass::moller_bhabha, mb_pair->first);

        // Fetch vector of materials
        const auto& mb_micro_xs = mb_pair->second;

        // Expect 2 materials
        EXPECT_EQ(2, mb_micro_xs.size());

        // Second material is stainless steel, with 3 elements
        const auto& steel_physvec_map = mb_micro_xs.back();
        EXPECT_EQ(3, steel_physvec_map.size());

        std::vector<double> element_id_list;
        std::vector<double> element_physvec_x_size, element_physvec_y_size;
        std::vector<double> element_physvec_x_front, element_physvec_y_front;
        std::vector<double> element_physvec_x_back, element_physvec_y_back;
        for (const auto& pair : steel_physvec_map)
        {
            const auto& phys_vec = pair.second;

            element_id_list.push_back(pair.first);

            element_physvec_x_size.push_back(phys_vec.x.size());
            element_physvec_y_size.push_back(phys_vec.y.size());

            element_physvec_x_front.push_back(phys_vec.x.front());
            element_physvec_y_front.push_back(phys_vec.y.front());

            element_physvec_x_back.push_back(phys_vec.x.back());
            element_physvec_y_back.push_back(phys_vec.y.back());
        }

        static const double expected_element_id_list[] = {0, 1, 2};

        // expected_element_physvec_y_size is not needed: x.size() == y.size()
        static const double expected_element_physvec_x_size[] = {9, 9, 9};

        static const double expected_element_physvec_x_front[]
            = {1.31345289979559, 1.31345289979559, 1.31345289979559};
        static const double expected_element_physvec_y_front[]
            = {4.57705e-24, 4.22497e-24, 4.92913e-24};

        static const double expected_element_physvec_x_back[]
            = {1e+08, 1e+08, 1e+08};
        static const double expected_element_physvec_y_back[] = {
            5.04687252343649e-24, 4.65865156009522e-24, 5.43509348677776e-24};

        EXPECT_VEC_EQ(expected_element_id_list, element_id_list);
        EXPECT_VEC_EQ(element_physvec_x_size, element_physvec_y_size);
        EXPECT_VEC_SOFT_EQ(expected_element_physvec_x_size,
                           element_physvec_x_size);

        EXPECT_VEC_SOFT_EQ(expected_element_physvec_x_front,
                           element_physvec_x_front);
        EXPECT_VEC_SOFT_EQ(expected_element_physvec_y_front,
                           element_physvec_y_front);

        EXPECT_VEC_SOFT_EQ(expected_element_physvec_x_back,
                           element_physvec_x_back);
        EXPECT_VEC_SOFT_EQ(expected_element_physvec_y_back,
                           element_physvec_y_back);

        for (auto i : celeritas::range(3))
        {
            EXPECT_GT(element_physvec_x_back[i], 0);
            EXPECT_GT(element_physvec_x_front[i], 0);
            EXPECT_GT(element_physvec_y_back[i], 0);
            EXPECT_GT(element_physvec_y_front[i], 0);
        }
    }
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, volumes)
{
    const auto volumes = data_.volumes;
    EXPECT_EQ(5, volumes.size());

    std::vector<unsigned int> material_ids;
    std::vector<std::string>  names, solids;

    for (const auto& volume : volumes)
    {
        material_ids.push_back(volume.material_id);
        names.push_back(volume.name);
        solids.push_back(volume.solid_name);
    }

    const unsigned int expected_material_ids[] = {1, 1, 1, 1, 0};
    const std::string  expected_names[]
        = {"box", "boxReplica", "boxReplica", "boxReplica", "World"};
    const std::string expected_solids[]
        = {"box", "boxReplica", "boxReplica2", "boxReplica3", "World"};

    EXPECT_VEC_EQ(expected_material_ids, material_ids);
    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_solids, solids);
}
