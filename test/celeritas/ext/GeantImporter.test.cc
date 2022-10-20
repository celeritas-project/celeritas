//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantImporter.hh"

#include "celeritas_config.h"
#include "corecel/io/StringUtils.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "celeritas_cmake_strings.h"
#include "celeritas_test.hh"
#if CELERITAS_USE_JSON
#    include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Helper functions
namespace
{
template<class Iter>
std::vector<std::string> to_vec_string(Iter iter, Iter end)
{
    std::vector<std::string> result;
    for (; iter != end; ++iter)
    {
        result.push_back(to_cstring(*iter));
    }
    return result;
}

bool geant4_is_v10()
{
    static const bool result = starts_with(celeritas_geant4_version, "10.");
    return result;
}
} // namespace

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeantImporterTest : public Test
{
  protected:
    using DataSelection = GeantImporter::DataSelection;

    struct ImportSummary
    {
        std::vector<std::string> particles;
        std::vector<std::string> processes;
        std::vector<std::string> models;

        void print_expected() const;
    };

    struct ImportXsSummary
    {
        std::vector<size_type> size;
        std::vector<real_type> x_bounds;
        std::vector<real_type> y_bounds;

        void print_expected() const;
    };

    ImportData import_geant(const DataSelection& selection)
    {
        // Only allow a single importer per global execution, because of Geant4
        // limitations.
        static GeantImporter import_geant(this->setup_geant());

        return import_geant(selection);
    }

    ImportSummary summarize(const ImportData& data) const;
    ImportXsSummary
    summarize(const ImportProcess::ElementPhysicsVectors& xs) const;

    virtual GeantSetup setup_geant() = 0;
};

//---------------------------------------------------------------------------//
auto GeantImporterTest::summarize(const ImportData& data) const -> ImportSummary
{
    ImportSummary s;
    for (const auto& p : data.particles)
    {
        s.particles.push_back(p.name);
    }

    // Create sorted unique set of process and model names inserted
    std::set<ImportProcessClass> pclass;
    std::set<ImportModelClass>   mclass;
    for (const auto& p : data.processes)
    {
        pclass.insert(p.process_class);
        mclass.insert(p.models.begin(), p.models.end());
    }
    s.processes = to_vec_string(pclass.begin(), pclass.end());
    s.models    = to_vec_string(mclass.begin(), mclass.end());
    return s;
}

void GeantImporterTest::ImportSummary::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static const char* expected_particles[] = "
         << repr(this->particles) << ";\n"
         << "EXPECT_VEC_EQ(expected_particles, summary.particles);\n"
            "static const char* expected_processes[] = "
         << repr(this->processes) << ";\n"
         << "EXPECT_VEC_EQ(expected_processes, summary.processes);\n"
            "static const char* expected_models[] = "
         << repr(this->models) << ";\n"
         << "EXPECT_VEC_EQ(expected_models, summary.models);\n"
            "/*** END CODE ***/\n";
}

auto GeantImporterTest::summarize(
    const ImportProcess::ElementPhysicsVectors& xs) const -> ImportXsSummary
{
    ImportXsSummary result;
    for (const auto& vec : xs)
    {
        EXPECT_FALSE(vec.x.empty());
        EXPECT_EQ(vec.x.size(), vec.y.size());
        result.size.push_back(vec.x.size());
        result.x_bounds.push_back(vec.x.front());
        result.x_bounds.push_back(vec.x.back());
        result.y_bounds.push_back(vec.y.front() / units::barn);
        result.y_bounds.push_back(vec.y.back() / units::barn);
    }
    return result;
}

void GeantImporterTest::ImportXsSummary::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
         << "static const real_type expected_size[] = " << repr(this->size)
         << ";\n"
         << "EXPECT_VEC_EQ(expected_size, result.size);\n"
         << "static const real_type expected_x_bounds[] = "
         << repr(this->x_bounds) << ";\n"
         << "EXPECT_VEC_SOFT_EQ(expected_x_bounds, result.x_bounds);\n"
         << "static const real_type expected_y_bounds[] = "
         << repr(this->y_bounds) << ";\n"
         << "EXPECT_VEC_SOFT_EQ(expected_y_bounds, result.y_bounds);\n"
         << "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
class FourSteelSlabsEmStandard : public GeantImporterTest
{
    GeantSetup setup_geant() override
    {
        GeantSetup::Options opts;
#if CELERITAS_USE_JSON
        {
            nlohmann::json    out = opts;
            static const char expected[]
                = R"json({"brems":"all","coulomb_scattering":false,"eloss_fluctuation":true,"em_bins_per_decade":7,"integral_approach":true,"linear_loss_limit":0.01,"lpm":true,"max_energy":[100000000.0,"MeV"],"min_energy":[0.0001,"MeV"],"msc":"urban","rayleigh_scattering":true})json";
            EXPECT_EQ(std::string(expected), std::string(out.dump()));
        }
#endif
        return GeantSetup(
            this->test_data_path("celeritas", "four-steel-slabs.gdml"), opts);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FourSteelSlabsEmStandard, em_particles)
{
    DataSelection options;
    options.particles = DataSelection::em;

    auto imported = this->import_geant(options);
    auto summary  = this->summarize(imported);

    static const char* expected_particles[] = {"e+", "e-", "gamma"};
    EXPECT_VEC_EQ(expected_particles, summary.particles);
    static const char* expected_processes[] = {"msc",
                                               "e_ioni",
                                               "e_brems",
                                               "photoelectric",
                                               "compton",
                                               "conversion",
                                               "rayleigh"};
    EXPECT_VEC_EQ(expected_processes, summary.processes);
    static const char* expected_models[] = {"urban_msc",
                                            "moller_bhabha",
                                            "e_brems_sb",
                                            "e_brems_lpm",
                                            "livermore_photoelectric",
                                            "klein_nishina",
                                            "bethe_heitler_lpm",
                                            "livermore_rayleigh"};
    EXPECT_VEC_EQ(expected_models, summary.models);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, em_hadronic)
{
    DataSelection options;
    options.particles = DataSelection::em | DataSelection::hadron;
    options.processes = DataSelection::em;

    auto imported = this->import_geant(options);
    auto summary  = this->summarize(imported);

    static const char* expected_particles[] = {"e+", "e-", "gamma", "proton"};
    EXPECT_VEC_EQ(expected_particles, summary.particles);
    static const char* expected_processes[] = {"msc",
                                               "e_ioni",
                                               "e_brems",
                                               "photoelectric",
                                               "compton",
                                               "conversion",
                                               "rayleigh"};
    EXPECT_VEC_EQ(expected_processes, summary.processes);
    static const char* expected_models[] = {"urban_msc",
                                            "moller_bhabha",
                                            "e_brems_sb",
                                            "e_brems_lpm",
                                            "livermore_photoelectric",
                                            "klein_nishina",
                                            "bethe_heitler_lpm",
                                            "livermore_rayleigh"};
    EXPECT_VEC_EQ(expected_models, summary.models);
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
    EXPECT_VEC_NEAR(expected_cutoff_energies, cutoff_energies,
                    geant4_is_v10() ? 1e-12 : 0.02);
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

    // Some values change substantially between v10 and v11.
    const real_type tol = geant4_is_v10() ? 1e-12 : 5e-3;

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
        EXPECT_SOFT_NEAR(839.66835335480653, steel.y.front(), tol);
        EXPECT_SOFT_NEAR(11.380485677836065, steel.y.back(), tol);
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
        EXPECT_SOFT_NEAR(2.3818927937550707e-07, steel.y.front(), tol);
        EXPECT_SOFT_NEAR(8786971.3079055995, steel.y.back(), tol);
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
        EXPECT_SOFT_NEAR(2.6269057995911775, steel.x.front(), tol);
        EXPECT_SOFT_EQ(1e8, steel.x.back());
        EXPECT_SOFT_EQ(0, steel.y.front());
        EXPECT_SOFT_NEAR(0.18987862452122845, steel.y[1], tol);
        EXPECT_SOFT_NEAR(0.43566778103861714, steel.y.back(), tol);
    }
    {
        // Test model microscopic cross sections
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

            // 2 materials; second material is stainless steel with 3
            // elements
            EXPECT_EQ(2, sb->second.size());
            EXPECT_EQ(3, sb->second.back().size());

            auto result = summarize(sb->second.back());

            static const real_type expected_size[]     = {5ul, 5ul, 5ul};
            static const real_type expected_x_bounds[] = {0.020923172565831,
                                                          1000,
                                                          0.020923172565831,
                                                          1000,
                                                          0.020923172565831,
                                                          1000};
            static const real_type expected_y_bounds[] = {19.855602934384,
                                                          77.270585225307,
                                                          16.824420929076,
                                                          66.692872575545,
                                                          23.159721368813,
                                                          88.395455128585};
            EXPECT_VEC_EQ(expected_size, result.size);
            EXPECT_VEC_NEAR(expected_x_bounds, result.x_bounds, tol);
            EXPECT_VEC_NEAR(expected_y_bounds, result.y_bounds, tol);
        }
        {
            // Check relativistic brems electron micro xs
            const auto& rb = brem->micro_xs.find(brem->models[1]);
            EXPECT_EQ(ImportModelClass::e_brems_lpm, rb->first);

            auto result = summarize(rb->second.back());

            static const real_type expected_size[] = {5ul, 5ul, 5ul};
            static const real_type expected_x_bounds[]
                = {1000, 100000000, 1000, 100000000, 1000, 100000000};
            static const real_type expected_y_bounds[] = {77.085320789881,
                                                          14.346956760121,
                                                          66.446696755766,
                                                          12.347642615031,
                                                          88.447643447573,
                                                          16.486026316006};

            EXPECT_VEC_EQ(expected_size, result.size);
            EXPECT_VEC_SOFT_EQ(expected_x_bounds, result.x_bounds);
            EXPECT_VEC_NEAR(expected_y_bounds, result.y_bounds, tol);
        }
        {
            // Check Bethe-Heitler micro xs
            auto conv = find_process(celeritas::pdg::gamma(),
                                     ImportProcessClass::conversion);
            ASSERT_NE(processes.end(), conv);
            EXPECT_EQ(celeritas::pdg::electron().get(), conv->secondary_pdg);
            EXPECT_EQ(1, conv->micro_xs.size());
            EXPECT_EQ(conv->models.size(), conv->micro_xs.size());

            const auto& bh = conv->micro_xs.find(conv->models[0]);
            EXPECT_EQ(ImportModelClass::bethe_heitler_lpm, bh->first);

            auto result = summarize(bh->second.back());

            static const unsigned int expected_size[]     = {9u, 9u, 9u};
            static const double       expected_x_bounds[] = {1.02199782,
                                                             100000000,
                                                             1.02199782,
                                                             100000000,
                                                             1.02199782,
                                                             100000000};
            static const double       expected_y_bounds[] = {1.4603666285612,
                                                             4.4976609946794,
                                                             1.250617083013,
                                                             3.8760336885145,
                                                             1.6856988385825,
                                                             5.1617257552977};

            EXPECT_VEC_EQ(expected_size, result.size);
            EXPECT_VEC_SOFT_EQ(expected_x_bounds, result.x_bounds);
            EXPECT_VEC_SOFT_EQ(expected_y_bounds, result.y_bounds);
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
    EXPECT_EQ(true, em_params.energy_loss_fluct);
    EXPECT_EQ(true, em_params.lpm);
    EXPECT_EQ(true, em_params.integral_approach);
    EXPECT_DOUBLE_EQ(0.01, em_params.linear_loss_limit);
    EXPECT_EQ(7, em_params.bins_per_decade);
    EXPECT_DOUBLE_EQ(1e-4, em_params.min_table_energy);
    EXPECT_DOUBLE_EQ(100e6, em_params.max_table_energy);
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
//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
