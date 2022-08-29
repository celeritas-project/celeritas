//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/ImportData.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/io/ImportData.hh"

#include <algorithm>
#include <unordered_map>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSetup.hh"
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
 * \note
 * G4EMLOW7.12 and G4EMLOW7.13 produce slightly different physics vector
 * values for steel, failing \c processes test.
 */
class ImportDataTest : public celeritas_test::Test
{
  protected:
    using DataSelection = GeantImporter::DataSelection;

    ImportData import_geant(const DataSelection& selection)
    {
        // Only allow a single importer per global execution
        static GeantImporter importer(this->setup_geant());
        return importer(selection);
    }

    virtual GeantSetup setup_geant() = 0;
};

//---------------------------------------------------------------------------//
class FourSteelSlabsEmStandard : public ImportDataTest
{
  public:
    GeantSetup setup_geant() override
    {
        GeantSetup::Options opts;
        return GeantSetup(
            this->test_data_path("celeritas", "four-steel-slabs.gdml"), opts);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, particles)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& particles = import_data.particles;
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
    static const char* expected_loaded_names[] = {"e+", "e-", "gamma"};
    static const int   expected_loaded_pdgs[]  = {-11, 11, 22};

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, elements)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& elements = import_data.elements;
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

    static const char*  expected_names[]          = {"Fe", "Cr", "Ni", "H"};
    static const int    expected_atomic_numbers[] = {26, 24, 28, 1};
    static const double expected_atomic_masses[]
        = {55.845110798, 51.996130137, 58.6933251009, 1.007940752665}; // [AMU]
    static const double expected_coulomb_factors[] = {0.04197339849163,
                                                      0.03592322294658,
                                                      0.04844802666907,
                                                      6.400838295295e-05};
    // Check inverse radiation length since soft equal comparison is
    // useless for extremely small values
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
TEST_F(FourSteelSlabsEmStandard, materials)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& materials = import_data.materials;
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
TEST_F(FourSteelSlabsEmStandard, processes)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& processes = import_data.processes;
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

            // 2 materials; seccond material is stainless steel with 3
            // elements
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
TEST_F(FourSteelSlabsEmStandard, volumes)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& volumes = import_data.volumes;
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

    static const char* expected_names[] = {"box0x125555be0",
                                           "box0x125556d20",
                                           "box0x125557160",
                                           "box0x1255575a0",
                                           "World0x125555f10"};

    static const char* expected_solids[] = {"box0x125555b70",
                                            "box0x125556c70",
                                            "box0x1255570a0",
                                            "box0x125557500",
                                            "World0x125555ea0"};

    EXPECT_VEC_EQ(expected_material_ids, material_ids);
    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_solids, solids);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, em_parameters)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& em_params = import_data.em_params;
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
TEST_F(FourSteelSlabsEmStandard, sb_data)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& sb_map = import_data.sb_data;
    EXPECT_EQ(4, sb_map.size());

    std::vector<int>    atomic_numbers;
    std::vector<double> sb_table_x;
    std::vector<double> sb_table_y;
    std::vector<double> sb_table_value;

    for (const auto& key : sb_map)
    {
        atomic_numbers.push_back(key.first);

        const auto& sb_table = key.second;
        sb_table_x.push_back(sb_table.x.front());
        sb_table_y.push_back(sb_table.y.front());
        sb_table_value.push_back(sb_table.value.front());
        sb_table_x.push_back(sb_table.x.back());
        sb_table_y.push_back(sb_table.y.back());
        sb_table_value.push_back(sb_table.value.back());
    }

    const int    expected_atomic_numbers[] = {1, 24, 26, 28};
    const double expected_sb_table_x[]
        = {-6.9078, 9.2103, -6.9078, 9.2103, -6.9078, 9.2103, -6.9078, 9.2103};
    const double expected_sb_table_y[]
        = {1e-12, 1, 1e-12, 1, 1e-12, 1, 1e-12, 1};
    const double expected_sb_table_value[] = {7.85327,
                                              0.046875,
                                              2.33528,
                                              0.717773,
                                              2.18202,
                                              0.748535,
                                              2.05115,
                                              0.776611};

    EXPECT_VEC_EQ(expected_atomic_numbers, atomic_numbers);
    EXPECT_VEC_EQ(expected_sb_table_x, sb_table_x);
    EXPECT_VEC_EQ(expected_sb_table_y, sb_table_y);
    EXPECT_VEC_EQ(expected_sb_table_value, sb_table_value);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, livermore_pe_data)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& lpe_map = import_data.livermore_pe_data;
    EXPECT_EQ(4, lpe_map.size());

    std::vector<int>    atomic_numbers;
    std::vector<size_t> shell_sizes;
    std::vector<double> thresh_lo;
    std::vector<double> thresh_hi;

    std::vector<double> shell_binding_energy;
    std::vector<double> shell_xs;
    std::vector<double> shell_energy;

    for (const auto& key : lpe_map)
    {
        atomic_numbers.push_back(key.first);

        const auto& ilpe = key.second;

        shell_sizes.push_back(ilpe.shells.size());

        const auto& shells_front = ilpe.shells.front();
        const auto& shells_back  = ilpe.shells.back();

        thresh_lo.push_back(ilpe.thresh_lo);
        thresh_hi.push_back(ilpe.thresh_hi);

        shell_binding_energy.push_back(shells_front.binding_energy);
        shell_binding_energy.push_back(shells_back.binding_energy);

        shell_xs.push_back(shells_front.xs.front());
        shell_xs.push_back(shells_front.xs.back());
        shell_energy.push_back(shells_front.energy.front());
        shell_energy.push_back(shells_front.energy.back());

        shell_xs.push_back(shells_back.xs.front());
        shell_xs.push_back(shells_back.xs.back());
        shell_energy.push_back(shells_back.energy.front());
        shell_energy.push_back(shells_back.energy.back());
    }

    const int           expected_atomic_numbers[] = {1, 24, 26, 28};
    const unsigned long expected_shell_sizes[]    = {1ul, 10ul, 10ul, 10ul};
    const double        expected_thresh_lo[]
        = {0.00537032, 0.00615, 0.0070834, 0.0083028};
    const double expected_thresh_hi[]
        = {0.0609537, 0.0616595, 0.0616595, 0.0595662};

    const double expected_shell_binding_energy[] = {1.361e-05,
                                                    1.361e-05,
                                                    0.0059576,
                                                    5.96e-06,
                                                    0.0070834,
                                                    7.53e-06,
                                                    0.0083028,
                                                    8.09e-06};

    const double expected_shell_xs[] = {1.58971e-08,
                                        1.6898e-09,
                                        1.58971e-08,
                                        1.6898e-09,
                                        0.00839767,
                                        0.0122729,
                                        1.39553e-10,
                                        4.05087e-06,
                                        0.0119194,
                                        0.0173188,
                                        7.35358e-10,
                                        1.46397e-05,
                                        0.0162052,
                                        0.0237477,
                                        1.20169e-09,
                                        1.91543e-05};

    const double expected_shell_energy[] = {1.361e-05,
                                            0.0933254,
                                            1.361e-05,
                                            0.0933254,
                                            0.0059576,
                                            0.0831764,
                                            5.96e-06,
                                            0.0630957,
                                            0.0070834,
                                            0.081283,
                                            7.53e-06,
                                            0.0653131,
                                            0.0083028,
                                            0.0776247,
                                            8.09e-06,
                                            0.0676083};

    EXPECT_VEC_EQ(expected_atomic_numbers, atomic_numbers);
    EXPECT_VEC_EQ(expected_shell_sizes, shell_sizes);
    EXPECT_VEC_SOFT_EQ(expected_thresh_lo, thresh_lo);
    EXPECT_VEC_SOFT_EQ(expected_thresh_hi, thresh_hi);
    EXPECT_VEC_SOFT_EQ(expected_shell_binding_energy, shell_binding_energy);
    EXPECT_VEC_SOFT_EQ(expected_shell_xs, shell_xs);
    EXPECT_VEC_SOFT_EQ(expected_shell_energy, shell_energy);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, atomic_relaxation_data)
{
    DataSelection options;
    options.particles = DataSelection::em;
    options.processes = DataSelection::em;
    auto import_data  = this->import_geant(options);

    const auto& ar_map = import_data.atomic_relaxation_data;
    EXPECT_EQ(4, ar_map.size());

    std::vector<int>    atomic_numbers;
    std::vector<size_t> shell_sizes;
    std::vector<int>    designator;
    std::vector<double> auger_probability;
    std::vector<double> auger_energy;
    std::vector<double> fluor_probability;
    std::vector<double> fluor_energy;

    for (const auto& key : ar_map)
    {
        atomic_numbers.push_back(key.first);

        const auto& shells = key.second.shells;
        shell_sizes.push_back(shells.size());

        if (shells.empty())
        {
            continue;
        }

        const auto& shells_front = shells.front();
        const auto& shells_back  = shells.back();

        designator.push_back(shells_front.designator);
        designator.push_back(shells_back.designator);

        auger_probability.push_back(shells_front.auger.front().probability);
        auger_probability.push_back(shells_front.auger.back().probability);
        auger_probability.push_back(shells_back.auger.front().probability);
        auger_probability.push_back(shells_back.auger.back().probability);
        auger_energy.push_back(shells_front.auger.front().energy);
        auger_energy.push_back(shells_front.auger.back().energy);
        auger_energy.push_back(shells_back.auger.front().energy);
        auger_energy.push_back(shells_back.auger.back().energy);

        fluor_probability.push_back(shells_front.fluor.front().probability);
        fluor_probability.push_back(shells_front.fluor.back().probability);
        fluor_probability.push_back(shells_back.fluor.front().probability);
        fluor_probability.push_back(shells_back.fluor.back().probability);
        fluor_energy.push_back(shells_front.fluor.front().energy);
        fluor_energy.push_back(shells_front.fluor.back().energy);
        fluor_energy.push_back(shells_back.fluor.front().energy);
        fluor_energy.push_back(shells_back.fluor.back().energy);
    }

    const int           expected_atomic_numbers[] = {1, 24, 26, 28};
    const unsigned long expected_shell_sizes[]    = {0ul, 7ul, 7ul, 7ul};
    const int           expected_designator[]     = {1, 11, 1, 11, 1, 11};

    const double expected_auger_probability[] = {0.048963695828293,
                                                 2.787499762505e-06,
                                                 0.015819909422702,
                                                 0.047183428103535,
                                                 0.044703908588515,
                                                 3.5127206748639e-06,
                                                 0.018361911975474,
                                                 0.076360349801533,
                                                 0.040678795307701,
                                                 3.1360396382578e-06,
                                                 0.021880812772728,
                                                 0.057510033570965};

    const double expected_auger_energy[] = {0.00458292,
                                            0.00594477,
                                            3.728e-05,
                                            3.787e-05,
                                            0.00539748,
                                            0.00706313,
                                            4.063e-05,
                                            4.618e-05,
                                            0.0062898,
                                            0.00828005,
                                            4.837e-05,
                                            5.546e-05};

    const double expected_fluor_probability[] = {0.082575892964534,
                                                 3.6954996851434e-06,
                                                 6.8993041093842e-08,
                                                 1.9834011813594e-08,
                                                 0.10139101947924,
                                                 8.7722616853269e-06,
                                                 3.4925922778373e-07,
                                                 1.158600755629e-07,
                                                 0.12105998603573,
                                                 1.8444997872369e-05,
                                                 1.0946006389633e-06,
                                                 5.1065929809277e-07};

    const double expected_fluor_energy[] = {0.00536786,
                                            0.00595123,
                                            4.374e-05,
                                            4.424e-05,
                                            0.00634985,
                                            0.00707066,
                                            5.354e-05,
                                            5.892e-05,
                                            0.00741782,
                                            0.00828814,
                                            6.329e-05,
                                            7.012e-05};

    EXPECT_VEC_EQ(expected_atomic_numbers, atomic_numbers);
    EXPECT_VEC_EQ(expected_shell_sizes, shell_sizes);
    EXPECT_VEC_EQ(expected_designator, designator);
    EXPECT_VEC_SOFT_EQ(expected_auger_probability, auger_probability);
    EXPECT_VEC_SOFT_EQ(expected_auger_energy, auger_energy);
    EXPECT_VEC_SOFT_EQ(expected_fluor_probability, fluor_probability);
    EXPECT_VEC_SOFT_EQ(expected_fluor_energy, fluor_energy);
}