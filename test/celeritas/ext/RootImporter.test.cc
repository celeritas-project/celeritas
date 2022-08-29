//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/RootImporter.hh"

#include <algorithm>
#include <unordered_map>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
/*!
 * The \e four-steel-slabs.root is created by the \e app/celer-export-geant
 * using the \e four-steel-slabs.gdml example file available in \e app/data .
 *
 * \note
 * G4EMLOW7.12 and G4EMLOW7.13 produce slightly different physics vector
 * values for steel, failing \c processes test.
 */
class RootImporterTest : public celeritas_test::Test
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
    const auto& particles = data_.particles;
    EXPECT_EQ(3, particles.size());

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
    static const char* expected_loaded_names[] = {"e+", "e-", "gamma"};

    static const int expected_loaded_pdgs[] = {-11, 11, 22};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, elements)
{
    const auto& elements = data_.elements;
    EXPECT_EQ(4, elements.size());

    std::vector<std::string> names;
    std::vector<int>         atomic_numbers;
    std::vector<double> atomic_masses, inv_rad_lengths_tsai, coulomb_factors;

    for (const auto& element : elements)
    {
        names.push_back(element.name);
        atomic_masses.push_back(element.atomic_mass);
        atomic_numbers.push_back(element.atomic_number);
        coulomb_factors.push_back(element.coulomb_factor);
        inv_rad_lengths_tsai.push_back(1 / element.radiation_length_tsai);
    }

    static const char* expected_names[] = {"Fe", "Cr", "Ni", "H"};
    static const int         expected_atomic_numbers[] = {26, 24, 28, 1};
    static const double      expected_atomic_masses[]
        = {55.845110798, 51.996130137, 58.6933251009, 1.007940752665}; // [AMU]
    static const double expected_coulomb_factors[] = {0.04197339849163,
                                                      0.03592322294658,
                                                      0.04844802666907,
                                                      6.400838295295e-05};
    // Check inverse radiation length since soft equal comparison is useless
    // for extremely small values
    static const double expected_inv_rad_lengths_tsai[] = {9.3141768784882e+40,
                                                           1.0803147822537e+41,
                                                           8.1192652842163e+40,
                                                           2.3509634762707e+43};

    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_atomic_numbers, atomic_numbers);
    EXPECT_VEC_SOFT_EQ(expected_atomic_masses, atomic_masses);
    EXPECT_VEC_SOFT_EQ(expected_coulomb_factors, coulomb_factors);
    EXPECT_VEC_SOFT_EQ(expected_inv_rad_lengths_tsai, inv_rad_lengths_tsai);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, materials)
{
    const auto& materials = data_.materials;
    EXPECT_EQ(2, materials.size());

    std::vector<std::string> names;
    std::vector<int>         states;
    std::vector<int>         pdgs;
    std::vector<double>      cutoff_energies, cutoff_ranges;
    std::vector<double> el_comps_ids, el_comps_mass_frac, el_comps_num_fracs;
    std::vector<double> densities, num_densities, e_densities, temperatures,
        rad_lengths, nuc_int_lengths;

    for (const auto& material : materials)
    {
        names.push_back(material.name);
        states.push_back((int)material.state);
        densities.push_back(material.density);
        e_densities.push_back(material.electron_density);
        num_densities.push_back(material.number_density);
        nuc_int_lengths.push_back(material.nuclear_int_length);
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

    static const char* expected_names[] = {"G4_Galactic", "G4_STAINLESS-STEEL"};
    EXPECT_VEC_EQ(expected_names, names);
    static const int expected_states[] = {3, 1};
    EXPECT_VEC_EQ(expected_states, states);
    static const int expected_pdgs[] = {-11, 11, 22, -11, 11, 22};
    EXPECT_VEC_EQ(expected_pdgs, pdgs);
    static const double expected_cutoff_energies[] = {0.00099,
                                                      0.00099,
                                                      0.00099,
                                                      1.22808845964606,
                                                      1.31345289979559,
                                                      0.0209231725658313};
    EXPECT_VEC_SOFT_EQ(expected_cutoff_energies, cutoff_energies);
    static const double expected_cutoff_ranges[]
        = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    EXPECT_VEC_SOFT_EQ(expected_cutoff_ranges, cutoff_ranges);
    static const double expected_densities[] = {1e-25, 8};
    EXPECT_VEC_SOFT_EQ(expected_densities, densities);
    static const double expected_e_densities[]
        = {0.05974697167543, 2.244432022882e+24};
    EXPECT_VEC_SOFT_EQ(expected_e_densities, e_densities);
    static const double expected_num_densities[]
        = {0.05974697167543, 8.699348925899e+22};
    EXPECT_VEC_SOFT_EQ(expected_num_densities, num_densities);
    static const double expected_nuc_int_lengths[]
        = {3.500000280825e+26, 16.67805709739};
    EXPECT_VEC_SOFT_EQ(expected_nuc_int_lengths, nuc_int_lengths);
    static const double expected_rad_lengths[]
        = {6.304350904227e+26, 1.738067064483};
    EXPECT_VEC_SOFT_EQ(expected_rad_lengths, rad_lengths);
    static const double expected_temperatures[] = {2.73, 293.15};
    EXPECT_VEC_SOFT_EQ(expected_temperatures, temperatures);
    static const double expected_el_comps_ids[] = {3, 0, 1, 2};
    EXPECT_VEC_SOFT_EQ(expected_el_comps_ids, el_comps_ids);
    static const double expected_el_comps_mass_frac[]
        = {1, 0.7462128746215, 0.1690010443115, 0.08478608106695};
    EXPECT_VEC_SOFT_EQ(expected_el_comps_mass_frac, el_comps_mass_frac);
    static const double expected_el_comps_num_fracs[] = {1, 0.74, 0.18, 0.08};
    EXPECT_VEC_SOFT_EQ(expected_el_comps_num_fracs, el_comps_num_fracs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, processes)
{
    const auto& processes = data_.processes;

    auto find_process = [&processes](PDGNumber pdg, ImportProcessClass ipc) {
        return std::find_if(processes.begin(),
                            processes.end(),
                            [&pdg, &ipc](const ImportProcess& proc) {
                                return PDGNumber{proc.particle_pdg} == pdg
                                       && proc.process_class == ipc;
                            });
    };

    auto ioni
        = find_process(celeritas::pdg::electron(), ImportProcessClass::e_ioni);
    ASSERT_NE(processes.end(), ioni);

    EXPECT_EQ(ImportProcessType::electromagnetic, ioni->process_type);
    ASSERT_EQ(1, ioni->models.size());
    EXPECT_EQ(ImportModelClass::moller_bhabha, ioni->models.front());
    EXPECT_EQ(celeritas::pdg::electron().get(), ioni->secondary_pdg);

    // No ionization micro xs
    EXPECT_EQ(0, ioni->micro_xs.size());

    const auto& tables = ioni->tables;
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
        // Test model microscopic cross sections
        auto get_values = [](const ImportProcess::ElementPhysicsVectorMap& xs) {
            std::unordered_map<std::string, std::vector<double>> result;
            for (const auto& kv : xs)
            {
                const auto& vec = kv.second;
                result["x_size"].push_back(vec.x.size());
                result["y_size"].push_back(vec.y.size());
                result["x_front"].push_back(vec.x.front());
                result["y_front"].push_back(vec.y.front() / units::barn);
                result["x_back"].push_back(vec.x.back());
                result["y_back"].push_back(vec.y.back() / units::barn);
                result["element_id"].push_back(kv.first);
            }
            return result;
        };

        auto brem = find_process(celeritas::pdg::electron(),
                                 ImportProcessClass::e_brems);
        ASSERT_NE(processes.end(), brem);
        EXPECT_EQ(celeritas::pdg::gamma().get(), brem->secondary_pdg);
        EXPECT_EQ(2, brem->micro_xs.size());
        EXPECT_EQ(brem->models.size(), brem->micro_xs.size());
        {
            // Check Seltzer-Berger electron micro xs
            const auto& sb = brem->micro_xs.find(brem->models[0]);
            EXPECT_EQ(ImportModelClass::e_brems_sb, sb->first);

            // 2 materials; seccond material is stainless steel with 3 elements
            EXPECT_EQ(2, sb->second.size());
            EXPECT_EQ(3, sb->second.back().size());

            static const double expected_size[] = {5, 5, 5};
            static const double expected_x_front[]
                = {0.0209231725658313, 0.0209231725658313, 0.0209231725658313};
            static const double expected_y_front[]
                = {19.855602934384, 16.824420929076, 23.159721368813};
            static const double expected_x_back[] = {1000, 1000, 1000};
            static const double expected_y_back[]
                = {77.270585225307, 66.692872575545, 88.395455128585};
            static const double expected_element_id[] = {0, 1, 2};

            auto actual = get_values(sb->second.back());
            EXPECT_VEC_EQ(expected_size, actual["x_size"]);
            EXPECT_VEC_EQ(expected_size, actual["y_size"]);
            EXPECT_VEC_SOFT_EQ(expected_x_front, actual["x_front"]);
            EXPECT_VEC_SOFT_EQ(expected_y_front, actual["y_front"]);
            EXPECT_VEC_SOFT_EQ(expected_x_back, actual["x_back"]);
            EXPECT_VEC_SOFT_EQ(expected_y_back, actual["y_back"]);
            EXPECT_VEC_EQ(expected_element_id, actual["element_id"]);
        }
        {
            // Check relativistic brems electron micro xs
            const auto& rb = brem->micro_xs.find(brem->models[1]);
            EXPECT_EQ(ImportModelClass::e_brems_lpm, rb->first);

            static const double expected_size[]    = {5, 5, 5};
            static const double expected_x_front[] = {1000, 1000, 1000};
            static const double expected_y_front[]
                = {77.085320789881, 66.446696755766, 88.447643447573};
            static const double expected_x_back[]
                = {100000000, 100000000, 100000000};
            static const double expected_y_back[]
                = {14.346956760121, 12.347642615031, 16.486026316006};
            static const double expected_element_id[] = {0, 1, 2};

            auto actual = get_values(rb->second.back());
            EXPECT_VEC_EQ(expected_size, actual["x_size"]);
            EXPECT_VEC_EQ(expected_size, actual["y_size"]);
            EXPECT_VEC_SOFT_EQ(expected_x_front, actual["x_front"]);
            EXPECT_VEC_SOFT_EQ(expected_y_front, actual["y_front"]);
            EXPECT_VEC_SOFT_EQ(expected_x_back, actual["x_back"]);
            EXPECT_VEC_SOFT_EQ(expected_y_back, actual["y_back"]);
            EXPECT_VEC_EQ(expected_element_id, actual["element_id"]);
        }
        {
            // Check Klein-Nishina micro xs
            auto comp = find_process(celeritas::pdg::gamma(),
                                     ImportProcessClass::compton);
            ASSERT_NE(processes.end(), comp);
            EXPECT_EQ(celeritas::pdg::electron().get(), comp->secondary_pdg);
            EXPECT_EQ(1, comp->micro_xs.size());
            EXPECT_EQ(comp->models.size(), comp->micro_xs.size());

            const auto& kn = comp->micro_xs.find(comp->models[0]);
            EXPECT_EQ(ImportModelClass::klein_nishina, kn->first);

            static const double expected_size[]    = {13, 13, 13};
            static const double expected_x_front[] = {0.0001, 0.0001, 0.0001};
            static const double expected_y_front[]
                = {1.0069880589339, 0.96395721121544, 1.042982687407};
            static const double expected_x_back[]
                = {100000000, 100000000, 100000000};
            static const double expected_y_back[] = {
                7.3005460134493e-07, 6.7387221120147e-07, 7.8623296376253e-07};
            static const double expected_element_id[] = {0, 1, 2};

            auto actual = get_values(kn->second.back());
            EXPECT_VEC_EQ(expected_size, actual["x_size"]);
            EXPECT_VEC_EQ(expected_size, actual["y_size"]);
            EXPECT_VEC_SOFT_EQ(expected_x_front, actual["x_front"]);
            EXPECT_VEC_SOFT_EQ(expected_y_front, actual["y_front"]);
            EXPECT_VEC_SOFT_EQ(expected_x_back, actual["x_back"]);
            EXPECT_VEC_SOFT_EQ(expected_y_back, actual["y_back"]);
            EXPECT_VEC_EQ(expected_element_id, actual["element_id"]);
        }
    }
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, volumes)
{
    const auto& volumes = data_.volumes;
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

    static const std::string expected_names[]  = {"box0x125555be0",
                                                 "box0x125556d20",
                                                 "box0x125557160",
                                                 "box0x1255575a0",
                                                 "World0x125555f10"};
    static const std::string expected_solids[] = {"box0x125555b70",
                                                  "box0x125556c70",
                                                  "box0x1255570a0",
                                                  "box0x125557500",
                                                  "World0x125555ea0"};

    EXPECT_VEC_EQ(expected_material_ids, material_ids);
    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_solids, solids);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, em_params)
{
    const auto& em_params = data_.em_params;
    EXPECT_EQ(7, em_params.size());

    std::vector<std::string> enum_string;
    std::vector<double>      value;

    for (const auto& key : em_params)
    {
        enum_string.push_back(to_cstring(key.first));
        value.push_back(key.second);
    }

    static const std::string expected_enum_string[] = {"energy_loss_fluct",
                                                       "lpm",
                                                       "integral_approach",
                                                       "linear_loss_limit",
                                                       "bins_per_decade",
                                                       "min_table_energy",
                                                       "max_table_energy"};

    static const double expected_value[]
        = {true, true, true, 0.01, 7, 1e-4, 100e6};

    EXPECT_VEC_EQ(expected_enum_string, enum_string);
    EXPECT_VEC_EQ(expected_value, value);
}
//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
