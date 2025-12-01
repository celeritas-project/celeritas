//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantImporter.hh"

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Version.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Helper functions
namespace
{

using namespace celeritas::units;

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

real_type to_inv_cm(real_type v)
{
    return native_value_to<InvCmXs>(v).value();
}

real_type to_sec(real_type v)
{
    return native_value_to<RealQuantity<Second>>(v).value();
}

auto const geant4_version = Version::from_string(cmake::geant4_version);
}  // namespace

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeantImporterTest : public GeantTestBase
{
  protected:
    using DataSelection = GeantImportDataSelection;
    using VecModelMaterial = std::vector<ImportModelMaterial>;

    struct ImportSummary
    {
        std::vector<std::string> particles;
        std::vector<std::string> processes;
        std::vector<std::string> models;

        void print_expected() const;
    };

    struct ImportXsSummary
    {
        std::vector<size_type> size;  //!< Number of micro XS in each material
        std::vector<real_type> energy;
        std::vector<real_type> xs;

        void print_expected() const;
    };

    ImportSummary summarize(ImportData const& data) const;
    ImportXsSummary summarize(VecModelMaterial const& xs) const;

    // Import data potentially with different selection options
    GeantImportDataSelection build_import_data_selection() const final
    {
        return selection_;
    }

    ImportProcess const&
    find_process(PDGNumber pdg, ImportProcessClass ipc) const
    {
        auto const& processes = this->imported_data().processes;
        auto result = std::find_if(processes.begin(),
                                   processes.end(),
                                   [pdg, ipc](ImportProcess const& proc) {
                                       return PDGNumber{proc.particle_pdg}
                                                  == pdg
                                              && proc.process_class == ipc;
                                   });
        CELER_VALIDATE(result != processes.end(),
                       << "missing process " << to_cstring(ipc)
                       << " for particle PDG=" << pdg.get());
        return *result;
    }

    ImportMscModel const&
    find_msc_model(PDGNumber pdg, ImportModelClass imc) const
    {
        auto const& models = this->imported_data().msc_models;
        auto result = std::find_if(
            models.begin(), models.end(), [pdg, imc](ImportMscModel const& m) {
                return PDGNumber{m.particle_pdg} == pdg && m.model_class == imc;
            });
        CELER_VALIDATE(result != models.end(),
                       << "missing model " << to_cstring(imc)
                       << " for particle PDG=" << pdg.get());
        return *result;
    }

    real_type comparison_tolerance() const
    {
        if (geant4_version != Version(11, 0, 3))
        {
            // Some values change substantially between geant versions
            return 5e-3;
        }
        if (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE)
        {
            // Single-precision unit constants cause single-precision
            // differences from reference
            return 1e-6;
        }
        return 1e-12;
    }

  protected:
    GeantImportDataSelection selection_{};
};

//---------------------------------------------------------------------------//
auto GeantImporterTest::summarize(ImportData const& data) const -> ImportSummary
{
    ImportSummary s;
    for (auto const& p : data.particles)
    {
        s.particles.push_back(p.name);
    }

    // Create sorted unique set of process and model names inserted
    std::set<ImportProcessClass> pclass;
    std::set<ImportModelClass> mclass;
    for (auto const& p : data.processes)
    {
        pclass.insert(p.process_class);
        for (auto const& m : p.models)
        {
            mclass.insert(m.model_class);
        }
    }
    for (auto const& m : data.msc_models)
    {
        mclass.insert(m.model_class);
    }
    s.processes = to_vec_string(pclass.begin(), pclass.end());
    s.models = to_vec_string(mclass.begin(), mclass.end());
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

auto GeantImporterTest::summarize(VecModelMaterial const& materials) const
    -> ImportXsSummary
{
    ImportXsSummary result;
    for (auto const& mat : materials)
    {
        result.size.push_back(
            mat.micro_xs.empty() ? 0 : mat.micro_xs.front().y.size());
        result.energy.push_back(mat.energy[Bound::lo]);
        result.energy.push_back(mat.energy[Bound::hi]);
    }

    // Skip export of first material, which is usually vacuum
    std::size_t mat_idx = 0;
    for (auto const& xs_vec : materials[mat_idx].micro_xs)
    {
        EXPECT_EQ(result.size[mat_idx], xs_vec.y.size());
    }
    ++mat_idx;

    for (; mat_idx < materials.size(); ++mat_idx)
    {
        for (auto const& xs_vec : materials[mat_idx].micro_xs)
        {
            EXPECT_EQ(result.size[mat_idx], xs_vec.y.size());
            result.xs.push_back(xs_vec.y.front() / barn);
            result.xs.push_back(xs_vec.y.back() / barn);
        }
    }
    return result;
}

void GeantImporterTest::ImportXsSummary::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
         << "static size_type const expected_size[] = " << repr(this->size)
         << ";\n"
         << "EXPECT_VEC_EQ(expected_size, result.size);\n"
         << "static real_type const expected_e[] = " << repr(this->energy)
         << ";\n"
         << "EXPECT_VEC_SOFT_EQ(expected_e, result.energy);\n"
         << "static real_type const expected_xs[] = " << repr(this->xs)
         << ";\n"
         << "EXPECT_VEC_SOFT_EQ(expected_xs, result.xs);\n"
         << "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
class FourSteelSlabsEmStandard : public GeantImporterTest
{
    std::string_view gdml_basename() const override
    {
        return "four-steel-slabs"sv;
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        GeantPhysicsOptions opts;
        opts.relaxation = RelaxationSelection::all;
        opts.muon.ionization = true;
        opts.muon.bremsstrahlung = true;
        opts.muon.pair_production = true;
        opts.verbose = true;
        if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
        {
            nlohmann::json out = opts;
            out.erase("_version");
            EXPECT_JSON_EQ(
                R"json({"_format":"geant-physics","_units":"cgs","angle_limit_factor":1.0,"annihilation":true,"apply_cuts":false,"brems":"all","compton_scattering":true,"coulomb_scattering":false,"default_cutoff":0.1,"eloss_fluctuation":true,"em_bins_per_decade":7,"form_factor":"exponential","gamma_conversion":true,"gamma_general":false,"integral_approach":true,"ionization":true,"linear_loss_limit":0.01,"lowest_electron_energy":[0.001,"MeV"],"lowest_muhad_energy":[0.001,"MeV"],"lpm":true,"max_energy":[100000000.0,"MeV"],"min_energy":[0.0001,"MeV"],"msc":"urban","msc_displaced":true,"msc_lambda_limit":0.1,"msc_muhad_displaced":false,"msc_muhad_range_factor":0.2,"msc_muhad_step_algorithm":"minimal","msc_range_factor":0.04,"msc_safety_factor":0.6,"msc_step_algorithm":"safety","msc_theta_limit":3.141592653589793,"muon":{"bremsstrahlung":true,"coulomb":false,"ionization":true,"msc":"none","pair_production":true},"optical":null,"photoelectric":true,"rayleigh_scattering":true,"relaxation":"all","seltzer_berger_limit":[1000.0,"MeV"],"verbose":true})json",
                std::string(out.dump()));
        }
        return opts;
    }
};

//---------------------------------------------------------------------------//
class TestEm3 : public GeantImporterTest
{
  protected:
    std::string_view gdml_basename() const override
    {
        return "testem3-flat"sv;
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        GeantPhysicsOptions opts;
        opts.relaxation = RelaxationSelection::none;
        opts.rayleigh_scattering = false;
        opts.verbose = false;
        return opts;
    }
};

//---------------------------------------------------------------------------//
class OneSteelSphere : public GeantImporterTest
{
  protected:
    std::string_view gdml_basename() const override
    {
        return "one-steel-sphere"sv;
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        GeantPhysicsOptions opts;
        opts.msc = MscModelSelection::urban_wentzelvi;
        opts.relaxation = RelaxationSelection::none;
        opts.verbose = false;
        return opts;
    }
};

//---------------------------------------------------------------------------//
class OneSteelSphereGG : public OneSteelSphere
{
  protected:
    void SetUp() override
    {
        if (geant4_version < Version{10, 6, 0})
        {
            GTEST_SKIP() << "Celeritas does not support gamma general for old "
                            "Geant4 versions";
        }
    }
    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = OneSteelSphere::build_geant_options();
        opts.gamma_general = true;
        opts.msc = MscModelSelection::urban;
        return opts;
    }
};

//---------------------------------------------------------------------------//
class LarSphere : public GeantImporterTest
{
  protected:
    std::string_view gdml_basename() const override { return "lar-sphere"sv; }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = GeantImporterTest::build_geant_options();
        opts.optical = {};
        CELER_ENSURE(opts.optical);
        return opts;
    }
};

//---------------------------------------------------------------------------//
class LarSphereExtramat : public GeantImporterTest
{
  protected:
    std::string_view gdml_basename() const override
    {
        return "lar-sphere-extramat"sv;
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = GeantImporterTest::build_geant_options();
        opts.optical = {};
        CELER_ENSURE(opts.optical);
        return opts;
    }
};

//---------------------------------------------------------------------------//
class OpticalSurfaces : public GeantImporterTest
{
  protected:
    std::string_view gdml_basename() const override
    {
        return "optical-surfaces"sv;
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = GeantImporterTest::build_geant_options();
        return opts;
    }
};

//---------------------------------------------------------------------------//
class Solids : public GeantImporterTest
{
  protected:
    std::string_view gdml_basename() const override { return "solids"sv; }

    GeantPhysicsOptions build_geant_options() const override
    {
        // only brems
        GeantPhysicsOptions opts;
        opts.compton_scattering = false;
        opts.coulomb_scattering = false;
        opts.photoelectric = false;
        opts.rayleigh_scattering = false;
        opts.gamma_conversion = false;
        opts.gamma_general = false;
        opts.ionization = false;
        opts.annihilation = false;
        opts.brems = BremsModelSelection::seltzer_berger;
        opts.msc = MscModelSelection::none;
        opts.relaxation = RelaxationSelection::none;
        opts.eloss_fluctuation = false;
        return opts;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(FourSteelSlabsEmStandard, em_particles)
{
    selection_.particles = DataSelection::em;

    auto&& imported = this->imported_data();
    auto summary = this->summarize(imported);

    static char const* expected_particles[]
        = {"e+", "e-", "gamma", "mu+", "mu-"};
    EXPECT_VEC_EQ(expected_particles, summary.particles);
    static char const* expected_processes[] = {
        "e_ioni",
        "e_brems",
        "photoelectric",
        "compton",
        "conversion",
        "rayleigh",
        "annihilation",
        "mu_ioni",
        "mu_brems",
        "mu_pair_prod",
    };
    EXPECT_VEC_EQ(expected_processes, summary.processes);
    static char const* expected_models[] = {
        "urban_msc",
        "icru_73_qo",
        "bragg",
        "moller_bhabha",
        "e_brems_sb",
        "e_brems_lpm",
        "e_plus_to_gg",
        "livermore_photoelectric",
        "klein_nishina",
        "bethe_heitler_lpm",
        "livermore_rayleigh",
        "mu_bethe_bloch",
        "mu_brems",
        "mu_pair_prod",
    };
    if (geant4_version < Version(11, 1, 0))
    {
        // Older versions of Geant4 use the Bethe-Bloch model for muon
        // ionization at intermediate energies
        auto iter = std::find(
            summary.models.begin(), summary.models.end(), "bethe_bloch");
        EXPECT_TRUE(iter != summary.models.end());
        summary.models.erase(iter, iter + 1);
    }
    EXPECT_VEC_EQ(expected_models, summary.models);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, em_hadronic)
{
    selection_.particles = DataSelection::em | DataSelection::hadron;
    selection_.processes = DataSelection::em;

    auto&& imported = this->imported_data();
    auto summary = this->summarize(imported);

    static char const* expected_particles[]
        = {"e+", "e-", "gamma", "mu+", "mu-", "proton"};
    EXPECT_VEC_EQ(expected_particles, summary.particles);
    static char const* expected_processes[] = {
        "e_ioni",
        "e_brems",
        "photoelectric",
        "compton",
        "conversion",
        "rayleigh",
        "annihilation",
        "mu_ioni",
        "mu_brems",
        "mu_pair_prod",
    };
    EXPECT_VEC_EQ(expected_processes, summary.processes);
    static char const* expected_models[] = {
        "urban_msc",
        "icru_73_qo",
        "bragg",
        "moller_bhabha",
        "e_brems_sb",
        "e_brems_lpm",
        "e_plus_to_gg",
        "livermore_photoelectric",
        "klein_nishina",
        "bethe_heitler_lpm",
        "livermore_rayleigh",
        "mu_bethe_bloch",
        "mu_brems",
        "mu_pair_prod",
    };
    if (geant4_version < Version(11, 1, 0))
    {
        // Older versions of Geant4 use the Bethe-Bloch model for muon
        // ionization at intermediate energies
        auto iter = std::find(
            summary.models.begin(), summary.models.end(), "bethe_bloch");
        EXPECT_TRUE(iter != summary.models.end());
        summary.models.erase(iter, iter + 1);
    }
    EXPECT_VEC_EQ(expected_models, summary.models);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, elements)
{
    auto&& import_data = this->imported_data();

    auto const& elements = import_data.elements;
    auto const& isotopes = import_data.isotopes;
    EXPECT_EQ(4, elements.size());

    std::vector<std::string> names;
    std::vector<int> atomic_numbers;
    std::vector<double> atomic_masses;
    std::vector<std::string> el_isotope_labels;
    std::vector<double> el_isotope_fractions;

    for (auto const& element : elements)
    {
        names.push_back(element.name);
        atomic_masses.push_back(element.atomic_mass);
        atomic_numbers.push_back(element.atomic_number);

        for (auto const& key : element.isotopes_fractions)
        {
            el_isotope_labels.push_back(isotopes[key.first].name);
            el_isotope_fractions.push_back(key.second);
        }
    }

    // clang-format off
    static std::string const expected_el_isotope_labels[]
        = {"Fe54", "Fe56", "Fe57", "Fe58", "Cr50", "Cr52", "Cr53", "Cr54",
            "Ni58", "Ni60", "Ni61", "Ni62", "Ni64", "H1", "H2"};
    // clang-format on

    static double const expected_el_isotope_fractions[] = {0.05845,
                                                           0.91754,
                                                           0.02119,
                                                           0.00282,
                                                           0.04345,
                                                           0.83789,
                                                           0.09501,
                                                           0.02365,
                                                           0.680769,
                                                           0.262231,
                                                           0.011399,
                                                           0.036345,
                                                           0.009256,
                                                           0.999885,
                                                           0.000115};

    static char const* expected_names[] = {"Fe", "Cr", "Ni", "H"};
    static int const expected_atomic_numbers[] = {26, 24, 28, 1};
    static double const expected_atomic_masses[] = {
        55.845110798, 51.996130137, 58.6933251009, 1.007940752665};  // [AMU]

    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_atomic_numbers, atomic_numbers);
    EXPECT_VEC_EQ(expected_el_isotope_labels, el_isotope_labels);
    EXPECT_VEC_SOFT_EQ(expected_atomic_masses, atomic_masses);
    EXPECT_VEC_SOFT_EQ(expected_el_isotope_fractions, el_isotope_fractions);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, isotopes)
{
    auto&& import_data = this->imported_data();
    auto const& isotopes = import_data.isotopes;

    std::vector<std::string> isotope_names;
    std::vector<int> isotope_atomic_number;
    std::vector<int> isotope_atomic_mass_number;
    std::vector<double> isotope_nuclear_mass;
    for (auto const& isotope : isotopes)
    {
        isotope_names.push_back(isotope.name);
        isotope_atomic_number.push_back(isotope.atomic_number);
        isotope_atomic_mass_number.push_back(isotope.atomic_mass_number);
        isotope_nuclear_mass.push_back(isotope.nuclear_mass);
    }

    // clang-format off
    static std::string const expected_isotope_names[]
        = {"Fe54", "Fe56", "Fe57", "Fe58", "Cr50", "Cr52", "Cr53", "Cr54",
            "Ni58", "Ni60", "Ni61", "Ni62", "Ni64", "H1", "H2"};

    static int const expected_isotope_atomic_number[]
        = {26, 26, 26, 26, 24, 24, 24, 24, 28, 28, 28, 28, 28, 1, 1};

    static int const expected_isotope_atomic_mass_number[]
        = {54, 56, 57, 58, 50, 52, 53, 54, 58, 60, 61, 62, 64, 1, 2};

    static double const expected_isotope_nuclear_mass[]
        = {50231.172508455, 52089.808009455, 53021.727279455, 53951.248020455,
            46512.204476826, 48370.036152826, 49301.662375826, 50231.508600826,
            53952.159103623, 55810.902779623, 56742.648018623, 57671.617505623,
            59534.252946623, 938.272013, 1875.6127932681};
    // clang-format on

    EXPECT_VEC_EQ(expected_isotope_names, isotope_names);
    EXPECT_VEC_EQ(expected_isotope_atomic_number, isotope_atomic_number);
    EXPECT_VEC_EQ(expected_isotope_atomic_mass_number,
                  isotope_atomic_mass_number);
    EXPECT_VEC_SOFT_EQ(expected_isotope_nuclear_mass, isotope_nuclear_mass);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, geo_materials)
{
    auto&& import_data = this->imported_data();

    auto const& materials = import_data.geo_materials;
    EXPECT_EQ(2, materials.size());

    std::vector<std::string> names;
    std::vector<int> states;
    std::vector<double> el_comps_ids, el_comps_num_fracs;
    std::vector<double> num_densities;
    std::vector<double> temperatures;

    for (auto const& material : materials)
    {
        names.push_back(material.name);
        states.push_back(static_cast<int>(material.state));
        num_densities.push_back(
            native_value_to<InvCcDensity>(material.number_density).value());
        temperatures.push_back(material.temperature);

        for (auto const& el_comp : material.elements)
        {
            el_comps_ids.push_back(el_comp.element_id);
            el_comps_num_fracs.push_back(el_comp.number_fraction);
        }
    }

    real_type const tol = this->comparison_tolerance();

    static char const* expected_names[] = {"G4_STAINLESS-STEEL", "G4_Galactic"};
    EXPECT_VEC_EQ(expected_names, names);
    static int const expected_states[] = {1, 3};
    EXPECT_VEC_EQ(expected_states, states);
    static double const expected_num_densities[]
        = {8.699348925899e+22, 0.05974697167543};
    EXPECT_VEC_NEAR(expected_num_densities, num_densities, tol);
    static double const expected_temperatures[] = {
        293.15,
        2.73,
    };
    EXPECT_VEC_SOFT_EQ(expected_temperatures, temperatures);
    static double const expected_el_comps_ids[] = {0, 1, 2, 3};
    EXPECT_VEC_SOFT_EQ(expected_el_comps_ids, el_comps_ids);
    static double const expected_el_comps_num_fracs[] = {0.74, 0.18, 0.08, 1};
    EXPECT_VEC_SOFT_EQ(expected_el_comps_num_fracs, el_comps_num_fracs);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, phys_materials)
{
    auto&& import_data = this->imported_data();

    auto const& materials = import_data.phys_materials;
    EXPECT_EQ(2, materials.size());

    std::vector<unsigned int> geo_ids;
    std::vector<int> pdgs;
    std::vector<double> cutoff_energies, cutoff_ranges;

    for (auto const& material : materials)
    {
        for (auto const& key : material.pdg_cutoffs)
        {
            pdgs.push_back(key.first);
            cutoff_energies.push_back(key.second.energy);
            cutoff_ranges.push_back(to_cm(key.second.range));
        }
    }

    real_type const tol = this->comparison_tolerance();
    static int const expected_pdgs[] = {-11, 11, 22, -11, 11, 22};
    EXPECT_VEC_EQ(expected_pdgs, pdgs);
    static double const expected_cutoff_energies[] = {0.00099,
                                                      0.00099,
                                                      0.00099,
                                                      1.22808845964606,
                                                      1.31345289979559,
                                                      0.0209231725658313};
    EXPECT_VEC_NEAR(expected_cutoff_energies,
                    cutoff_energies,
                    geant4_version.major() == 10 ? 1e-12 : 0.02);
    static double const expected_cutoff_ranges[]
        = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    EXPECT_VEC_NEAR(expected_cutoff_ranges, cutoff_ranges, tol);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, eioni)
{
    real_type const tol = this->comparison_tolerance();

    ImportProcess const& proc = this->find_process(celeritas::pdg::electron(),
                                                   ImportProcessClass::e_ioni);
    EXPECT_EQ(ImportProcessType::electromagnetic, proc.process_type);
    EXPECT_EQ(celeritas::pdg::electron().get(), proc.secondary_pdg);
    EXPECT_FALSE(proc.applies_at_rest);

    // Test model
    ASSERT_EQ(1, proc.models.size());
    {
        auto const& model = proc.models[0];
        EXPECT_EQ(ImportModelClass::moller_bhabha, model.model_class);
        for (auto const& m : model.materials)
        {
            EXPECT_EQ(0, m.micro_xs.size());
        }
    }

    EXPECT_TRUE(proc.dedx);
    EXPECT_TRUE(proc.lambda);
    EXPECT_FALSE(proc.lambda_prim);
    {
        // Test energy loss table
        ImportPhysicsTable const& dedx = proc.dedx;
        EXPECT_EQ(ImportUnits::mev, dedx.x_units);
        EXPECT_EQ(ImportUnits::mev_per_len, dedx.y_units);
        ASSERT_EQ(2, dedx.grids.size());

        auto const& steel = dedx.grids.back();
        ASSERT_EQ(85, steel.y.size());
        EXPECT_SOFT_EQ(1e-4, std::exp(steel.x[Bound::lo]));
        EXPECT_SOFT_EQ(1e8, std::exp(steel.x[Bound::hi]));
        EXPECT_SOFT_NEAR(839.66835335480653, to_inv_cm(steel.y.front()), tol);
        EXPECT_SOFT_NEAR(11.378226755591747, to_inv_cm(steel.y.back()), tol);
    }
    {
        // Test cross-section table
        ImportPhysicsTable const& lambda = proc.lambda;
        EXPECT_EQ(ImportUnits::mev, lambda.x_units);
        EXPECT_EQ(ImportUnits::len_inv, lambda.y_units);
        ASSERT_EQ(2, lambda.grids.size());

        auto const& steel = lambda.grids.back();
        ASSERT_EQ(54, steel.y.size());
        EXPECT_SOFT_NEAR(2.616556310615175, std::exp(steel.x[Bound::lo]), tol);
        EXPECT_SOFT_EQ(1e8, std::exp(steel.x[Bound::hi]));
        EXPECT_SOFT_EQ(0, steel.y.front());
        EXPECT_SOFT_NEAR(0.1905939505829807, to_inv_cm(steel.y[1]), tol);
        EXPECT_SOFT_NEAR(0.4373910150880348, to_inv_cm(steel.y.back()), tol);
    }
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, ebrems)
{
    ImportProcess const& proc = this->find_process(
        celeritas::pdg::electron(), ImportProcessClass::e_brems);
    EXPECT_EQ(celeritas::pdg::gamma().get(), proc.secondary_pdg);
    EXPECT_FALSE(proc.applies_at_rest);
    ASSERT_EQ(2, proc.models.size());
    if (geant4_version < Version{11})
    {
        GTEST_SKIP() << "Cross sections changed with Geant4 version 11; older "
                        "versions are not checked";
    }

    {
        // Check Seltzer-Berger electron micro xs
        auto const& model = proc.models[0];
        EXPECT_EQ(ImportModelClass::e_brems_sb, model.model_class);
        EXPECT_EQ(2, model.materials.size());

        auto result = summarize(model.materials);
        static size_type const expected_size[] = {7ul, 5ul};
        EXPECT_VEC_EQ(expected_size, result.size);
        static real_type const expected_e[]
            = {0.001, 1000, 0.020822442086622, 1000};
        EXPECT_VEC_SOFT_EQ(expected_e, result.energy);
        static real_type const expected_xs[] = {19.90859573288,
                                                77.272184544415,
                                                16.869369978465,
                                                66.694254412524,
                                                23.221614672926,
                                                88.397283181803};
        EXPECT_VEC_SOFT_EQ(expected_xs, result.xs);
    }
    {
        // Check relativistic brems electron micro xs
        auto const& model = proc.models[1];
        EXPECT_EQ(ImportModelClass::e_brems_lpm, model.model_class);
        EXPECT_EQ(2, model.materials.size());

        auto result = summarize(model.materials);
        static size_type const expected_size[] = {6ul, 6ul};
        EXPECT_VEC_EQ(expected_size, result.size);
        static real_type const expected_e[]
            = {1000, 100000000, 1000, 100000000};
        EXPECT_VEC_SOFT_EQ(expected_e, result.energy);
        static real_type const expected_xs[] = {77.086886023111,
                                                14.346968386977,
                                                66.448046061979,
                                                12.347652116819,
                                                88.449439286966,
                                                16.486040161073};
        EXPECT_VEC_SOFT_EQ(expected_xs, result.xs);
    }
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, conv)
{
    ImportProcess const& proc = this->find_process(
        celeritas::pdg::gamma(), ImportProcessClass::conversion);
    EXPECT_EQ(celeritas::pdg::electron().get(), proc.secondary_pdg);
    EXPECT_FALSE(proc.applies_at_rest);
    ASSERT_EQ(1, proc.models.size());

    {
        // Check Bethe-Heitler micro xs
        auto const& model = proc.models[0];
        EXPECT_EQ(ImportModelClass::bethe_heitler_lpm, model.model_class);

        EXPECT_EQ(2, model.materials.size());

        auto result = summarize(model.materials);

        static size_type const expected_size[] = {9ul, 9ul};
        EXPECT_VEC_EQ(expected_size, result.size);
        static real_type const expected_e[]
            = {1.02199782, 100000000, 1.02199782, 100000000};
        EXPECT_VEC_SOFT_EQ(expected_e, result.energy);
        static real_type const expected_xs[] = {1.4603666285612,
                                                4.4976609946794,
                                                1.250617083013,
                                                3.8760336885145,
                                                1.6856988385825,
                                                5.1617257552977};
        EXPECT_VEC_SOFT_EQ(expected_xs, result.xs);
    }
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, anni)
{
    ImportProcess const& proc = this->find_process(
        celeritas::pdg::positron(), ImportProcessClass::annihilation);
    EXPECT_EQ(celeritas::pdg::gamma().get(), proc.secondary_pdg);
    EXPECT_TRUE(proc.applies_at_rest);
    ASSERT_EQ(1, proc.models.size());

    auto const& model = proc.models[0];
    EXPECT_EQ(ImportModelClass::e_plus_to_gg, model.model_class);

    EXPECT_EQ(2, model.materials.size());
    auto result = summarize(model.materials);
    static double const expected_energy[]
        = {0.0001, 100000000, 0.0001, 100000000};
    static unsigned int const expected_size[] = {0, 0};
    EXPECT_VEC_EQ(expected_size, result.size);
    EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
    EXPECT_TRUE(result.xs.empty());
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, muioni)
{
    real_type const tol = this->comparison_tolerance();

    ImportProcess const& mu_minus = this->find_process(
        celeritas::pdg::mu_minus(), ImportProcessClass::mu_ioni);
    EXPECT_EQ(ImportProcessType::electromagnetic, mu_minus.process_type);
    EXPECT_EQ(celeritas::pdg::electron().get(), mu_minus.secondary_pdg);
    EXPECT_FALSE(mu_minus.applies_at_rest);

    // Test model
    ASSERT_EQ(geant4_version < Version(11, 1, 0) ? 3 : 2,
              mu_minus.models.size());
    {
        auto const& model = mu_minus.models.front();
        EXPECT_EQ(ImportModelClass::icru_73_qo, model.model_class);

        auto result = summarize(model.materials);
        EXPECT_TRUE(result.xs.empty());
        static unsigned int const expected_size[] = {0, 0};
        EXPECT_VEC_EQ(expected_size, result.size);
        static double const expected_energy[] = {0.0001, 0.2, 0.0001, 0.2};
        EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
    }
    if (geant4_version < Version(11, 1, 0))
    {
        auto const& model = mu_minus.models[1];
        EXPECT_EQ(ImportModelClass::bethe_bloch, model.model_class);

        auto result = summarize(model.materials);
        EXPECT_TRUE(result.xs.empty());
        static unsigned int const expected_size[] = {0, 0};
        EXPECT_VEC_EQ(expected_size, result.size);
        static double const expected_energy[] = {0.2, 1000, 0.2, 1000};
        EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
    }
    {
        auto const& model = mu_minus.models.back();
        EXPECT_EQ(ImportModelClass::mu_bethe_bloch, model.model_class);

        auto result = summarize(model.materials);
        EXPECT_TRUE(result.xs.empty());
        static unsigned int const expected_size[] = {0, 0};
        EXPECT_VEC_EQ(expected_size, result.size);
        if (geant4_version < Version(11, 1, 0))
        {
            static double const expected_energy[]
                = {1000, 100000000, 1000, 100000000};
            EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
        }
        else
        {
            static double const expected_energy[]
                = {0.2, 100000000, 0.2, 100000000};
            EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
        }
    }

    EXPECT_TRUE(mu_minus.dedx);
    EXPECT_TRUE(mu_minus.lambda);
    EXPECT_FALSE(mu_minus.lambda_prim);
    {
        // Test energy loss table
        ImportPhysicsTable const& dedx = mu_minus.dedx;
        EXPECT_EQ(ImportUnits::mev, dedx.x_units);
        EXPECT_EQ(ImportUnits::mev_per_len, dedx.y_units);
        ASSERT_EQ(2, dedx.grids.size());

        auto const& steel = dedx.grids.back();
        ASSERT_EQ(85, steel.y.size());
        EXPECT_SOFT_EQ(1e-4, std::exp(steel.x[Bound::lo]));
        EXPECT_SOFT_EQ(1e8, std::exp(steel.x[Bound::hi]));
        EXPECT_SOFT_NEAR(83.221648535690946, to_inv_cm(steel.y.front()), tol);
        EXPECT_SOFT_NEAR(11.40198961519433, to_inv_cm(steel.y.back()), tol);
    }
    {
        // Test cross-section table
        ImportPhysicsTable const& xs = mu_minus.lambda;
        EXPECT_EQ(ImportUnits::mev, xs.x_units);
        EXPECT_EQ(ImportUnits::len_inv, xs.y_units);
        ASSERT_EQ(2, xs.grids.size());

        auto const& steel = xs.grids.back();
        ASSERT_EQ(45, steel.y.size());
        EXPECT_SOFT_NEAR(54.542938808612199, std::exp(steel.x[Bound::lo]), tol);
        EXPECT_SOFT_EQ(1e8, std::exp(steel.x[Bound::hi]));
        EXPECT_SOFT_EQ(0, steel.y.front());
        EXPECT_SOFT_NEAR(0.10167398809855273, to_inv_cm(steel.y[1]), tol);
        EXPECT_SOFT_NEAR(0.47315182268065914, to_inv_cm(steel.y.back()), tol);
    }

    // Check mu+
    ImportProcess const& mu_plus = this->find_process(
        celeritas::pdg::mu_plus(), ImportProcessClass::mu_ioni);
    EXPECT_EQ(ImportProcessType::electromagnetic, mu_plus.process_type);
    EXPECT_EQ(celeritas::pdg::electron().get(), mu_plus.secondary_pdg);
    EXPECT_FALSE(mu_plus.applies_at_rest);

    auto const& models = mu_plus.models;
    ASSERT_EQ(geant4_version < Version(11, 1, 0) ? 3 : 2, models.size());
    EXPECT_EQ(ImportModelClass::bragg, models.front().model_class);
    EXPECT_EQ(ImportModelClass::mu_bethe_bloch, models.back().model_class);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, volumes)
{
    auto&& import_data = this->imported_data();

    auto const& volumes = import_data.volumes;
    EXPECT_EQ(5, volumes.size());

    std::vector<unsigned int> material_ids;
    std::vector<std::string> names, solids;

    for (auto const& volume : volumes)
    {
        material_ids.push_back(volume.phys_material_id);
        names.push_back(volume.name);
        solids.push_back(volume.solid_name);
    }

    unsigned int const expected_material_ids[] = {1, 1, 1, 1, 0};
    static char const* expected_names[] = {
        "box@0",
        "box@1",
        "box@2",
        "box@3",
        "World",
    };
    static char const* expected_solids[] = {
        "box",
        "box",
        "box",
        "box",
        "World",
    };

    EXPECT_VEC_EQ(expected_material_ids, material_ids);
    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_solids, solids);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, em_parameters)
{
    auto&& import_data = this->imported_data();

    auto const& em_params = import_data.em_params;
    EXPECT_EQ(true, em_params.energy_loss_fluct);
    EXPECT_EQ(true, em_params.lpm);
    EXPECT_EQ(true, em_params.integral_approach);
    EXPECT_DOUBLE_EQ(0.01, em_params.linear_loss_limit);
    EXPECT_DOUBLE_EQ(0.001, em_params.lowest_electron_energy);
    EXPECT_EQ(true, em_params.auger);
    EXPECT_EQ(MscStepLimitAlgorithm::safety, em_params.msc_step_algorithm);
    EXPECT_DOUBLE_EQ(0.04, em_params.msc_range_factor);
    EXPECT_DOUBLE_EQ(0.6, em_params.msc_safety_factor);
    EXPECT_REAL_EQ(0.1, to_cm(em_params.msc_lambda_limit));
    EXPECT_DOUBLE_EQ(static_cast<double>(constants::pi),
                     em_params.msc_theta_limit);
    EXPECT_EQ(false, em_params.apply_cuts);
    EXPECT_EQ(1, em_params.screening_factor);
    EXPECT_EQ(1, em_params.angle_limit_factor);
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, trans_parameters)
{
    auto&& import_data = this->imported_data();

    EXPECT_EQ(1000, import_data.trans_params.max_substeps);
    EXPECT_EQ(5, import_data.trans_params.looping.size());
    for (auto const& kv : import_data.trans_params.looping)
    {
        EXPECT_EQ(10, kv.second.threshold_trials);
        EXPECT_EQ(250, kv.second.important_energy);
    }
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, sb_data)
{
    auto&& import_data = this->imported_data();

    auto const& sb_map = import_data.seltzer_berger.atomic_xs;
    EXPECT_EQ(4, sb_map.size());

    std::vector<int> atomic_numbers;
    std::vector<double> sb_table_x;
    std::vector<double> sb_table_y;
    std::vector<double> sb_table_value;

    for (auto const& key : sb_map)
    {
        atomic_numbers.push_back(key.first.get());

        auto const& sb_table = key.second;
        sb_table_x.push_back(sb_table.x.front());
        sb_table_y.push_back(sb_table.y.front());
        sb_table_value.push_back(sb_table.value.front());
        sb_table_x.push_back(sb_table.x.back());
        sb_table_y.push_back(sb_table.y.back());
        sb_table_value.push_back(sb_table.value.back());
    }

    int const expected_atomic_numbers[] = {1, 24, 26, 28};
    double const expected_sb_table_x[]
        = {-6.9078, 9.2103, -6.9078, 9.2103, -6.9078, 9.2103, -6.9078, 9.2103};
    double const expected_sb_table_y[]
        = {1e-12, 1, 1e-12, 1, 1e-12, 1, 1e-12, 1};
    double const expected_sb_table_value[] = {7.85327,
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
TEST_F(FourSteelSlabsEmStandard, mu_pair_production_data)
{
    auto&& import_data = this->imported_data();

    auto const& data = import_data.mu_production.muppet_table;

    using Z = AtomicNumber;
    Z const expected_atomic_number[] = {Z{1}, Z{4}, Z{13}, Z{29}, Z{92}};
    EXPECT_VEC_EQ(expected_atomic_number, data.atomic_number);

    EXPECT_EQ(5, data.grids.size());

    std::vector<double> table_x;
    std::vector<double> table_y;
    std::vector<double> table_value;

    for (auto const& pv : data.grids)
    {
        table_x.push_back(pv.x.front());
        table_y.push_back(pv.y.front());
        table_value.push_back(pv.value.front() / barn);
        table_x.push_back(pv.x.back());
        table_y.push_back(pv.y.back());
        table_value.push_back(pv.value.back() / barn);
    }

    real_type const tol = geant4_version < Version(11, 1, 0) ? 1e-12 : 0.03;

    static double const expected_table_x[] = {
        6.9077552789821,
        18.420680743952,
        6.9077552789821,
        18.420680743952,
        6.9077552789821,
        18.420680743952,
        6.9077552789821,
        18.420680743952,
        6.9077552789821,
        18.420680743952,
    };
    static double const expected_table_y[] = {
        -6.1928487397154,
        0,
        -6.1928487397154,
        0,
        -6.1928487397154,
        0,
        -6.1928487397154,
        0,
        -6.1928487397154,
        0,
    };
    static double const expected_table_value[] = {
        0,
        0.24363843626056,
        0,
        2.257683855817,
        0,
        18.983775898741,
        0,
        86.585529175975,
        0,
        793.41396760823,
    };
    EXPECT_VEC_NEAR(expected_table_x, table_x, tol);
    EXPECT_VEC_NEAR(expected_table_y, table_y, tol);
    EXPECT_VEC_NEAR(
        expected_table_value, table_value, this->comparison_tolerance());
}

//---------------------------------------------------------------------------//
TEST_F(FourSteelSlabsEmStandard, livermore_pe_data)
{
    ScopedLogStorer scoped_log{&celeritas::world_logger(), LogLevel::warning};
    auto&& import_data = this->imported_data();
    EXPECT_TRUE(scoped_log.empty()) << scoped_log;

    auto const& lpe_map = import_data.livermore_photo.atomic_xs;
    EXPECT_EQ(4, lpe_map.size());

    std::vector<int> atomic_numbers;
    std::vector<size_t> shell_sizes;
    std::vector<double> thresh_lo;
    std::vector<double> thresh_hi;

    std::vector<double> shell_binding_energy;
    std::vector<double> shell_xs;
    std::vector<double> shell_energy;

    for (auto const& key : lpe_map)
    {
        atomic_numbers.push_back(key.first.get());

        auto const& ilpe = key.second;

        shell_sizes.push_back(ilpe.shells.size());

        auto const& shells_front = ilpe.shells.front();
        auto const& shells_back = ilpe.shells.back();

        thresh_lo.push_back(ilpe.thresh_lo);
        thresh_hi.push_back(ilpe.thresh_hi);

        shell_binding_energy.push_back(shells_front.binding_energy);
        shell_binding_energy.push_back(shells_back.binding_energy);

        shell_xs.push_back(shells_front.xs.y.front());
        shell_xs.push_back(shells_front.xs.y.back());
        shell_energy.push_back(shells_front.xs.x.front());
        shell_energy.push_back(shells_front.xs.x.back());

        shell_xs.push_back(shells_back.xs.y.front());
        shell_xs.push_back(shells_back.xs.y.back());
        shell_energy.push_back(shells_back.xs.x.front());
        shell_energy.push_back(shells_back.xs.x.back());
    }

    int const expected_atomic_numbers[] = {1, 24, 26, 28};
    unsigned long const expected_shell_sizes[] = {1ul, 10ul, 10ul, 10ul};
    double const expected_thresh_lo[]
        = {0.00537032, 0.00615, 0.0070834, 0.0083028};
    double const expected_thresh_hi[]
        = {0.0609537, 0.0616595, 0.0616595, 0.0595662};

    double const expected_shell_binding_energy[] = {1.361e-05,
                                                    1.361e-05,
                                                    0.0059576,
                                                    5.96e-06,
                                                    0.0070834,
                                                    7.53e-06,
                                                    0.0083028,
                                                    8.09e-06};

    double const expected_shell_xs[] = {1.58971e-08,
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

    double const expected_shell_energy[] = {1.361e-05,
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
    auto&& import_data = this->imported_data();

    auto const& ar_map = import_data.atomic_relaxation.atomic_xs;
    EXPECT_EQ(4, ar_map.size());

    std::vector<int> atomic_numbers;
    std::vector<size_t> shell_sizes;
    std::vector<int> designator;
    std::vector<double> auger_probability;
    std::vector<double> auger_energy;
    std::vector<double> fluor_probability;
    std::vector<double> fluor_energy;

    for (auto const& key : ar_map)
    {
        atomic_numbers.push_back(key.first.get());

        auto const& shells = key.second.shells;
        shell_sizes.push_back(shells.size());

        if (shells.empty())
        {
            continue;
        }

        auto const& shells_front = shells.front();
        auto const& shells_back = shells.back();

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

    int const expected_atomic_numbers[] = {1, 24, 26, 28};
    unsigned long const expected_shell_sizes[] = {0ul, 7ul, 7ul, 7ul};
    int const expected_designator[] = {1, 11, 1, 11, 1, 11};

    double const expected_auger_probability[] = {0.048963695828293,
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

    double const expected_auger_energy[] = {0.00458292,
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

    double const expected_fluor_probability[] = {0.082575892964534,
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

    double const expected_fluor_energy[] = {0.00536786,
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

TEST_F(TestEm3, volume_names)
{
    selection_.reader_data = false;
    auto const& volumes = this->imported_data().volumes;

    std::vector<std::string> names;
    for (auto const& volume : volumes)
    {
        names.push_back(volume.name);
    }

    // clang-format off
    static std::string const expected_names[] = {"gap_0", "absorber_0", "gap_1", "absorber_1", "gap_2", "absorber_2", "gap_3", "absorber_3", "gap_4", "absorber_4", "gap_5", "absorber_5", "gap_6", "absorber_6", "gap_7", "absorber_7", "gap_8", "absorber_8", "gap_9", "absorber_9", "gap_10", "absorber_10", "gap_11", "absorber_11", "gap_12", "absorber_12", "gap_13", "absorber_13", "gap_14", "absorber_14", "gap_15", "absorber_15", "gap_16", "absorber_16", "gap_17", "absorber_17", "gap_18", "absorber_18", "gap_19", "absorber_19", "gap_20", "absorber_20", "gap_21", "absorber_21", "gap_22", "absorber_22", "gap_23", "absorber_23", "gap_24", "absorber_24", "gap_25", "absorber_25", "gap_26", "absorber_26", "gap_27", "absorber_27", "gap_28", "absorber_28", "gap_29", "absorber_29", "gap_30", "absorber_30", "gap_31", "absorber_31", "gap_32", "absorber_32", "gap_33", "absorber_33", "gap_34", "absorber_34", "gap_35", "absorber_35", "gap_36", "absorber_36", "gap_37", "absorber_37", "gap_38", "absorber_38", "gap_39", "absorber_39", "gap_40", "absorber_40", "gap_41", "absorber_41", "gap_42", "absorber_42", "gap_43", "absorber_43", "gap_44", "absorber_44", "gap_45", "absorber_45", "gap_46", "absorber_46", "gap_47", "absorber_47", "gap_48", "absorber_48", "gap_49", "absorber_49", "world"};
    // clang-format on
    EXPECT_VEC_EQ(expected_names, names);
}

//---------------------------------------------------------------------------//

TEST_F(TestEm3, unique_volumes)
{
    selection_.reader_data = false;
    selection_.unique_volumes = true;

    auto const& volumes = this->imported_data().volumes;

    EXPECT_EQ(101, volumes.size());
    EXPECT_EQ("gap_0", volumes.front().name)
        << "Front name: '" << volumes.front().name << "'";
}

//---------------------------------------------------------------------------//

TEST_F(OneSteelSphere, cutoffs)
{
    auto&& import_data = this->imported_data();

    EXPECT_EQ(2, import_data.volumes.size());
    EXPECT_EQ(2, import_data.phys_materials.size());

    // Check secondary production cuts
    std::vector<int> pdg;
    std::vector<double> range_cut, energy_cut;
    for (auto const& mat : import_data.phys_materials)
    {
        for (auto const& cut : mat.pdg_cutoffs)
        {
            pdg.push_back(cut.first);
            range_cut.push_back(to_cm(cut.second.range));
            energy_cut.push_back(cut.second.energy);
        }
    }
    static int const expected_pdg[] = {-11, 11, 22, -11, 11, 22};
    EXPECT_VEC_EQ(expected_pdg, pdg);
    // 1 mm range cut in vacuum, 50 m range cut in steel
    static real_type const expected_range_cut[]
        = {0.1, 0.1, 0.1, 5000, 5000, 5000};
    EXPECT_VEC_SOFT_EQ(expected_range_cut, range_cut);
    static double const expected_energy_cut[] = {0.00099,
                                                 0.00099,
                                                 0.00099,
                                                 9549.6516356879,
                                                 9549.6516356879,
                                                 9549.6516356879};
    EXPECT_VEC_SOFT_EQ(expected_energy_cut, energy_cut);
}

TEST_F(OneSteelSphere, physics)
{
    real_type const tol = this->comparison_tolerance();

    // Check the bremsstrahlung cross sections
    ImportProcess const& brems = this->find_process(
        celeritas::pdg::electron(), ImportProcessClass::e_brems);
    EXPECT_FALSE(brems.applies_at_rest);

    EXPECT_FALSE(brems.dedx);
    EXPECT_TRUE(brems.lambda);
    EXPECT_FALSE(brems.lambda_prim);
    ASSERT_EQ(2, brems.models.size());
    {
        // Check Seltzer-Berger electron micro xs
        auto const& model = brems.models[0];
        EXPECT_EQ(ImportModelClass::e_brems_sb, model.model_class);
        EXPECT_EQ(2, model.materials.size());

        auto result = summarize(model.materials);
        static unsigned int const expected_size[] = {7u, 0u};
        EXPECT_VEC_EQ(expected_size, result.size);
        static double const expected_energy[]
            = {0.001, 1000, 9549.6516356879, 1000};
        EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
        // Gamma production cut in steel is higher than the SB model upper
        // energy limit, so there will be no micro xs
        EXPECT_TRUE(result.xs.empty());
    }
    {
        // Check relativistic brems electron micro xs
        auto const& model = brems.models[1];
        EXPECT_EQ(ImportModelClass::e_brems_lpm, model.model_class);
        EXPECT_EQ(2, model.materials.size());

        auto result = summarize(model.materials);
        static size_type const expected_size[] = {6u, 5u};
        EXPECT_VEC_EQ(expected_size, result.size);
        static double const expected_energy[]
            = {1000, 100000000, 9549.6516356879, 100000000};
        EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
        static double const expected_xs[] = {16.197663688566,
                                             14.176435287746,
                                             13.963271396942,
                                             12.201090525228,
                                             18.583905773638,
                                             16.289792829097};
        EXPECT_VEC_SOFT_EQ(expected_xs, result.xs);
    }
    {
        // Check the bremsstrahlung macro xs
        ImportPhysicsTable const& xs = brems.lambda;
        ASSERT_EQ(2, xs.grids.size());
        auto const& steel = xs.grids.back();
        ASSERT_EQ(29, steel.y.size());
        EXPECT_SOFT_EQ(9549.651635687942, std::exp(steel.x[Bound::lo]));
        EXPECT_SOFT_EQ(1e8, std::exp(steel.x[Bound::hi]));
    }
    {
        // Check the ionization electron macro xs
        ImportProcess const& ioni = this->find_process(
            celeritas::pdg::electron(), ImportProcessClass::e_ioni);
        EXPECT_TRUE(ioni.dedx);
        EXPECT_TRUE(ioni.lambda);
        EXPECT_FALSE(ioni.lambda_prim);

        // Lambda table for steel
        ImportPhysicsTable const& xs = ioni.lambda;
        ASSERT_EQ(2, xs.grids.size());
        auto const& steel = xs.grids.back();
        ASSERT_EQ(27, steel.y.size());
        // Starts at min primary energy = 2 * electron production cut for
        // primary electrons
        EXPECT_SOFT_EQ(19099.303271375884, std::exp(steel.x[Bound::lo]));
        EXPECT_SOFT_EQ(1e8, std::exp(steel.x[Bound::hi]));
    }
    {
        // Check the ionization positron macro xs
        ImportProcess const& ioni = this->find_process(
            celeritas::pdg::positron(), ImportProcessClass::e_ioni);
        EXPECT_TRUE(ioni.dedx);
        EXPECT_TRUE(ioni.lambda);
        EXPECT_FALSE(ioni.lambda_prim);

        // Lambda table for steel
        ImportPhysicsTable const& xs = ioni.lambda;
        ASSERT_EQ(2, xs.grids.size());
        auto const& steel = xs.grids.back();
        ASSERT_EQ(29, steel.y.size());
        // Start at min primary energy = electron production cut for primary
        // positrons
        EXPECT_SOFT_EQ(9549.651635687942, std::exp(steel.x[Bound::lo]));
        EXPECT_SOFT_EQ(1e8, std::exp(steel.x[Bound::hi]));
    }
    {
        // Check Urban MSC bounds
        ImportMscModel const& msc = this->find_msc_model(
            celeritas::pdg::electron(), ImportModelClass::urban_msc);
        EXPECT_TRUE(msc);
        for (auto const& pv : msc.xs_table.grids)
        {
            ASSERT_TRUE(pv);
            EXPECT_SOFT_EQ(1e-4, std::exp(pv.x[Bound::lo]));
            EXPECT_SOFT_EQ(1e2, std::exp(pv.x[Bound::hi]));
        }
        auto const& steel = msc.xs_table.grids.back();
        EXPECT_SOFT_NEAR(0.23785296407525, to_inv_cm(steel.y.front()), tol);
        EXPECT_SOFT_NEAR(128.58803359467, to_inv_cm(steel.y.back()), tol);
    }
    {
        // Check Wentzel VI MSC bounds
        ImportMscModel const& msc = this->find_msc_model(
            celeritas::pdg::electron(), ImportModelClass::wentzel_vi_uni);
        EXPECT_TRUE(msc);
        for (auto const& pv : msc.xs_table.grids)
        {
            ASSERT_TRUE(pv);
            EXPECT_SOFT_EQ(1e2, std::exp(pv.x[Bound::lo]));
            EXPECT_SOFT_EQ(1e8, std::exp(pv.x[Bound::hi]));
        }
        auto const& steel = msc.xs_table.grids.back();
        EXPECT_SOFT_NEAR(114.93265072267, to_inv_cm(steel.y.front()), tol);
        EXPECT_SOFT_NEAR(116.59035766356, to_inv_cm(steel.y.back()), tol);
    }
}

TEST_F(OneSteelSphereGG, physics)
{
    auto&& imported = this->imported_data();
    auto summary = this->summarize(imported);

    static char const* expected_particles[] = {"e+", "e-", "gamma"};
    EXPECT_VEC_EQ(expected_particles, summary.particles);
    static char const* expected_processes[] = {"e_ioni",
                                               "e_brems",
                                               "photoelectric",
                                               "compton",
                                               "conversion",
                                               "rayleigh",
                                               "annihilation"};
    EXPECT_VEC_EQ(expected_processes, summary.processes);
    static char const* expected_models[] = {"urban_msc",
                                            "moller_bhabha",
                                            "e_brems_sb",
                                            "e_brems_lpm",
                                            "e_plus_to_gg",
                                            "livermore_photoelectric",
                                            "klein_nishina",
                                            "bethe_heitler_lpm",
                                            "livermore_rayleigh"};
    EXPECT_VEC_EQ(expected_models, summary.models);

    {
        // Check Urban MSC bounds
        ImportMscModel const& msc = this->find_msc_model(
            celeritas::pdg::electron(), ImportModelClass::urban_msc);
        EXPECT_TRUE(msc);
        for (auto const& pv : msc.xs_table.grids)
        {
            EXPECT_SOFT_EQ(1e-4, std::exp(pv.x[Bound::lo]));
            EXPECT_SOFT_EQ(1e8, std::exp(pv.x[Bound::hi]));
        }
    }
}

TEST_F(LarSphere, optical)
{
    ScopedLogStorer scoped_log{&celeritas::world_logger(), LogLevel::info};
    auto&& imported = this->imported_data();
    ASSERT_EQ(5, imported.optical_models.size());
    ASSERT_EQ(1, imported.optical_materials.size());
    ASSERT_EQ(3, imported.geo_materials.size());
    ASSERT_EQ(2, imported.phys_materials.size());

    // First material is vacuum, no optical properties
    ASSERT_EQ(0, imported.phys_materials[0].geo_material_id);
    EXPECT_EQ("vacuum", imported.geo_materials[0].name);
    EXPECT_EQ(ImportPhysMaterial::unspecified,
              imported.phys_materials[0].optical_material_id);

    // Second material is liquid argon
    ASSERT_EQ(1, imported.phys_materials[1].geo_material_id);
    EXPECT_EQ("lAr", imported.geo_materials[1].name);
    ASSERT_EQ(0, imported.phys_materials[1].optical_material_id);

    // Most optical properties in the geometry are pulled from the Geant4
    // example examples/advanced/CaTS/gdml/LArTPC.gdml

    // Check scintillation optical properties
    auto const& optical = imported.optical_materials[0];
    auto const& scint = optical.scintillation;
    EXPECT_TRUE(scint);

    // Material scintillation
    constexpr auto tol = SoftEqual<real_type>{}.rel();
    EXPECT_REAL_EQ(1, scint.resolution_scale);
    EXPECT_REAL_EQ(5000, scint.material.yield_per_energy);
    EXPECT_EQ(3, scint.material.components.size());
    std::vector<double> components;
    for (auto const& comp : scint.material.components)
    {
        components.push_back(comp.yield_frac);
        components.push_back(to_cm(comp.gauss.lambda_mean));
        components.push_back(to_cm(comp.gauss.lambda_sigma));
        components.push_back(to_sec(comp.rise_time));
        components.push_back(to_sec(comp.fall_time));
    }
    static double const expected_components[] = {
        3,
        1.28e-05,
        1e-06,
        1e-08,
        6e-09,
        1,
        1.28e-05,
        1e-06,
        1e-08,
        1.5e-06,
        1,
        0,
        0,
        1e-08,
        3e-06,
    };
    EXPECT_VEC_NEAR(expected_components, components, tol);

    // Particle scintillation
    EXPECT_EQ(6, scint.particles.size());
    std::vector<int> pdgs;
    std::vector<double> yield_vecs;
    std::vector<size_t> comp_sizes;
    std::vector<double> comp_y, comp_lm, comp_ls, comp_rt, comp_ft;
    for (auto const& iter : scint.particles)
    {
        pdgs.push_back(iter.first);
        auto const& part = iter.second;
        for (auto i : range(part.yield_vector.x.size()))
        {
            yield_vecs.push_back(part.yield_vector.x[i]);
            yield_vecs.push_back(part.yield_vector.y[i]);
        }
        comp_sizes.push_back(part.components.size());
        for (auto comp : part.components)
        {
            comp_y.push_back(comp.yield_frac);
            comp_lm.push_back(to_cm(comp.gauss.lambda_mean));
            comp_ls.push_back(to_cm(comp.gauss.lambda_sigma));
            comp_rt.push_back(to_sec(comp.rise_time));
            comp_ft.push_back(to_sec(comp.fall_time));
        }
    }
    static int const expected_pdgs[]
        = {11, 90, 2212, 1000010020, 1000010030, 1000020040};
    static double const expected_yield_vecs[] = {
        1e-06, 3750, 6, 5000,  // electron
        1e-06, 2000, 6, 4000,  // ion
        1e-06, 2500, 6, 4200,  // proton
        1e-06, 1200, 6, 3000,  // deuteron
        1e-06, 1500, 6, 3500,  // triton
        1e-06, 1700, 6, 3700  // alpha
    };
    EXPECT_VEC_EQ(expected_pdgs, pdgs);
    EXPECT_VEC_EQ(expected_yield_vecs, yield_vecs);

    // The electron has one component, the rest has no components
    static unsigned long const expected_comp_sizes[]
        = {1ul, 0ul, 0ul, 0ul, 0ul, 0ul};
    EXPECT_VEC_EQ(expected_comp_sizes, comp_sizes);

    // Electron component data
    static double const expected_comp_y[] = {4000};
    static double const expected_comp_lm[] = {1e-05};
    static double const expected_comp_ls[] = {1e-06};
    static double const expected_comp_rt[] = {1.5e-08};
    static double const expected_comp_ft[] = {5e-09};

    EXPECT_VEC_EQ(expected_comp_y, expected_comp_y);
    EXPECT_VEC_EQ(expected_comp_lm, expected_comp_lm);
    EXPECT_VEC_EQ(expected_comp_ls, expected_comp_ls);
    EXPECT_VEC_EQ(expected_comp_rt, expected_comp_rt);
    EXPECT_VEC_EQ(expected_comp_ft, expected_comp_ft);

    // Check Rayleigh optical properties
    auto const& rayleigh_model = imported.optical_models[1];
    EXPECT_EQ(optical::ImportModelClass::rayleigh, rayleigh_model.model_class);
    ASSERT_EQ(1, rayleigh_model.mfp_table.size());

    auto const& rayleigh_mfp = rayleigh_model.mfp_table.front();
    EXPECT_EQ(11, rayleigh_mfp.x.size());
    EXPECT_DOUBLE_EQ(1.55e-06, rayleigh_mfp.x.front());
    EXPECT_DOUBLE_EQ(1.55e-05, rayleigh_mfp.x.back());
    EXPECT_REAL_EQ(32142.9, to_cm(rayleigh_mfp.y.front()));
    EXPECT_REAL_EQ(54.6429, to_cm(rayleigh_mfp.y.back()));

    auto const& rayleigh_mat = optical.rayleigh;
    EXPECT_TRUE(rayleigh_mat);
    EXPECT_EQ(1, rayleigh_mat.scale_factor);
    EXPECT_REAL_EQ(0.024673059861887867 * centimeter * ipow<2>(second) / gram,
                   rayleigh_mat.compressibility);

    // Check absorption optical properties
    auto const& absorption_model = imported.optical_models[0];
    EXPECT_EQ(optical::ImportModelClass::absorption,
              absorption_model.model_class);
    ASSERT_EQ(1, absorption_model.mfp_table.size());

    auto const& absorption_mfp = absorption_model.mfp_table.front();
    EXPECT_EQ(2, absorption_mfp.x.size());
    EXPECT_DOUBLE_EQ(1.3778e-06, absorption_mfp.x.front());
    EXPECT_DOUBLE_EQ(1.55e-05, absorption_mfp.x.back());
    EXPECT_REAL_EQ(86.4473, to_cm(absorption_mfp.y.front()));
    EXPECT_REAL_EQ(0.000296154, to_cm(absorption_mfp.y.back()));

    {
        // Check WLS optical properties
        auto const& model = imported.optical_models[3];
        EXPECT_EQ(optical::ImportModelClass::wls, model.model_class);
        ASSERT_EQ(1, model.mfp_table.size());

        auto const& mfp = model.mfp_table.front();
        EXPECT_EQ(2, mfp.x.size());
        EXPECT_EQ(mfp.x.size(), mfp.y.size());

        auto const& mat = optical.wls;
        EXPECT_TRUE(mat);
        EXPECT_SOFT_EQ(0.456, mat.mean_num_photons);
        EXPECT_SOFT_EQ(6e-9, to_sec(mat.time_constant));

        std::vector<double> abslen_grid, comp_grid;
        for (auto i : range(mfp.x.size()))
        {
            abslen_grid.push_back(mfp.x[i]);
            abslen_grid.push_back(to_cm(mfp.y[i]));
            comp_grid.push_back(mat.component.x[i]);
            comp_grid.push_back(mat.component.y[i]);
        }

        static real_type const expected_abslen_grid[]
            = {1.3778e-06, 0.1, 1.55e-05, 0.01};
        static double const expected_comp_grid[]
            = {1.3778e-06, 0.1, 1e-05, 0.9};
        EXPECT_VEC_SOFT_EQ(expected_abslen_grid, abslen_grid);
        EXPECT_VEC_SOFT_EQ(expected_comp_grid, comp_grid);
    }
    {
        // Check WLS2 optical properties
        auto const& model = imported.optical_models[4];
        EXPECT_EQ(optical::ImportModelClass::wls2, model.model_class);
        ASSERT_EQ(1, model.mfp_table.size());

        auto const& mfp = model.mfp_table.front();
        EXPECT_EQ(2, mfp.x.size());
        EXPECT_EQ(mfp.x.size(), mfp.y.size());

        auto const& mat = optical.wls2;
        EXPECT_TRUE(mat);
        EXPECT_REAL_EQ(0.123, mat.mean_num_photons);
        EXPECT_REAL_EQ(6e-9, to_sec(mat.time_constant));

        std::vector<double> abslen_grid, comp_grid;
        for (auto i : range(mfp.x.size()))
        {
            abslen_grid.push_back(mfp.x[i]);
            abslen_grid.push_back(to_cm(mfp.y[i]));
            comp_grid.push_back(mat.component.x[i]);
            comp_grid.push_back(mat.component.y[i]);
        }

        static double const expected_abslen_grid[]
            = {1.3778e-06, 0.1, 1.55e-05, 0.01};
        static double const expected_comp_grid[]
            = {1.771e-06, 0.3, 2.484e-06, 0.8};
        EXPECT_VEC_NEAR(
            expected_abslen_grid, abslen_grid, this->comparison_tolerance());
        EXPECT_VEC_SOFT_EQ(expected_comp_grid, comp_grid);
    }

    // Check common optical properties
    // Refractive index data in the geometry comes from the refractive index
    // database https://refractiveindex.info and was calculating using the
    // methods described in: E. Grace, A. Butcher, J.  Monroe, J. A. Nikkel.
    // Index of refraction, Rayleigh scattering length, and Sellmeier
    // coefficients in solid and liquid argon and xenon, Nucl.  Instr. Meth.
    // Phys. Res. A 867, 204-208 (2017)
    auto const& properties = optical.properties;
    EXPECT_TRUE(properties);
    EXPECT_EQ(101, properties.refractive_index.x.size());
    EXPECT_DOUBLE_EQ(1.8785e-06, properties.refractive_index.x.front());
    EXPECT_DOUBLE_EQ(1.0597e-05, properties.refractive_index.x.back());
    EXPECT_DOUBLE_EQ(1.2221243542166, properties.refractive_index.y.front());
    EXPECT_DOUBLE_EQ(1.6167515615703, properties.refractive_index.y.back());
}

TEST_F(LarSphereExtramat, optical)
{
    auto&& imported = this->imported_data();
    ASSERT_EQ(5, imported.optical_models.size());
    ASSERT_EQ(1, imported.optical_materials.size());
    ASSERT_EQ(3, imported.geo_materials.size());
    ASSERT_EQ(2, imported.phys_materials.size());

    // First material is vacuum, no optical properties
    ASSERT_EQ(0, imported.phys_materials[0].geo_material_id);
    EXPECT_EQ("vacuum", imported.geo_materials[0].name);
    EXPECT_EQ(ImportPhysMaterial::unspecified,
              imported.phys_materials[0].optical_material_id);

    // Second material is liquid argon
    ASSERT_EQ(1, imported.phys_materials[1].geo_material_id);
    EXPECT_EQ("lAr", imported.geo_materials[1].name);
    ASSERT_EQ(0, imported.phys_materials[1].optical_material_id);

    // Check scintillation, WLS, and WLS2 optical properties
    auto const& optical = imported.optical_materials[0];
    EXPECT_FALSE(optical.scintillation);
    EXPECT_FALSE(optical.wls);
    EXPECT_FALSE(optical.wls2);

    // Check Rayleigh optical properties
    auto const& rayleigh_model = imported.optical_models[1];
    EXPECT_EQ(optical::ImportModelClass::rayleigh, rayleigh_model.model_class);
    ASSERT_EQ(1, rayleigh_model.mfp_table.size());

    auto const& rayleigh_mfp = rayleigh_model.mfp_table.front();
    EXPECT_EQ(2, rayleigh_mfp.x.size());
    EXPECT_DOUBLE_EQ(1.55e-06, rayleigh_mfp.x.front());
    EXPECT_DOUBLE_EQ(1.55e-05, rayleigh_mfp.x.back());
    EXPECT_REAL_EQ(32142.9, to_cm(rayleigh_mfp.y.front()));
    EXPECT_REAL_EQ(54.6429, to_cm(rayleigh_mfp.y.back()));

    // Check common optical properties
    // Refractive index data in the geometry comes from the refractive index
    // database https://refractiveindex.info and was calculating using the
    // methods described in: E. Grace, A. Butcher, J.  Monroe, J. A. Nikkel.
    // Index of refraction, Rayleigh scattering length, and Sellmeier
    // coefficients in solid and liquid argon and xenon, Nucl.  Instr. Meth.
    // Phys. Res. A 867, 204-208 (2017)
    auto const& properties = optical.properties;
    EXPECT_TRUE(properties);
    EXPECT_EQ(2, properties.refractive_index.x.size());
}

//---------------------------------------------------------------------------//

TEST_F(Solids, volumes_only)
{
    selection_.reader_data = false;
    selection_.particles = GeantImportDataSelection::none;
    selection_.processes = GeantImportDataSelection::none;
    selection_.materials = false;
    selection_.reader_data = false;
    selection_.unique_volumes = false;

    auto const& imported = this->imported_data();
    EXPECT_EQ(0, imported.processes.size());
    EXPECT_EQ(0, imported.particles.size());
    EXPECT_EQ(0, imported.elements.size());
    EXPECT_EQ(0, imported.geo_materials.size());
    EXPECT_EQ(0, imported.phys_materials.size());

    std::vector<std::string> names;
    for (auto const& volume : imported.volumes)
    {
        names.push_back(volume.name);
    }

    static char const* const expected_names[]
        = {"box500",     "cone1",    "para1",     "sphere1",    "parabol1",
           "trap1",      "trd1",     "trd2",      "",           "trd3_refl@1",
           "tube100",    "boolean1", "polycone1", "genPocone1", "ellipsoid1",
           "tetrah1",    "orb1",     "polyhedr1", "hype1",      "elltube1",
           "ellcone1",   "arb8b",    "arb8a",     "xtru1",      "World",
           "trd3_refl@0"};
    EXPECT_VEC_EQ(expected_names, names);
}

TEST_F(Solids, volumes_unique)
{
    selection_.reader_data = false;
    selection_.particles = GeantImportDataSelection::none;
    selection_.processes = GeantImportDataSelection::none;
    selection_.materials = false;
    selection_.reader_data = false;
    selection_.unique_volumes = true;  // emulates accel/SharedParams

    auto const& imported = this->imported_data();

    std::vector<std::string> names;
    for (auto const& volume : imported.volumes)
    {
        names.push_back(volume.name);
    }
    static char const* const expected_names[]
        = {"box500",     "cone1",    "para1",     "sphere1",    "parabol1",
           "trap1",      "trd1",     "trd2",      "",           "trd3_refl@1",
           "tube100",    "boolean1", "polycone1", "genPocone1", "ellipsoid1",
           "tetrah1",    "orb1",     "polyhedr1", "hype1",      "elltube1",
           "ellcone1",   "arb8b",    "arb8a",     "xtru1",      "World",
           "trd3_refl@0"};
    EXPECT_VEC_EQ(expected_names, names);
}

TEST_F(Solids, physics)
{
    selection_.reader_data = false;

    auto&& imported = this->imported_data();
    auto summary = this->summarize(imported);

    static char const* expected_particles[] = {"e+", "e-", "gamma"};
    EXPECT_VEC_EQ(expected_particles, summary.particles);
    static char const* expected_processes[] = {"e_brems"};
    EXPECT_VEC_EQ(expected_processes, summary.processes);
    static char const* expected_models[] = {"e_brems_sb"};
    EXPECT_VEC_EQ(expected_models, summary.models);
}

//---------------------------------------------------------------------------//

TEST_F(OpticalSurfaces, surfaces)
{
    auto specular_spike = [](inp::DielectricInteraction const& di) {
        return di.reflection
            .reflection_grids[optical::ReflectionMode::specular_spike];
    };

    auto specular_lobe = [](inp::DielectricInteraction const& di) {
        return di.reflection
            .reflection_grids[optical::ReflectionMode::specular_lobe];
    };

    auto backscatter = [](inp::DielectricInteraction const& di) {
        return di.reflection
            .reflection_grids[optical::ReflectionMode::backscatter];
    };

    auto&& osp = this->imported_data().optical_physics.surfaces;
    EXPECT_TRUE(osp);

#define OS_IS_MAPPED(MEMBER, ID) (osp.MEMBER.find(ID) != osp.MEMBER.end())

    // sphere_surf: glisur, polished, dielectric-dielectric, specular spike
    {
        PhysSurfaceId sid{0};
        EXPECT_TRUE(OS_IS_MAPPED(roughness.polished, sid));
        EXPECT_FALSE(OS_IS_MAPPED(roughness.smear, sid));
        EXPECT_FALSE(OS_IS_MAPPED(roughness.gaussian, sid));

        EXPECT_TRUE(OS_IS_MAPPED(reflectivity.grid, sid));
        EXPECT_FALSE(OS_IS_MAPPED(reflectivity.fresnel, sid));

        EXPECT_TRUE(OS_IS_MAPPED(interaction.dielectric, sid));

        auto& di = osp.interaction.dielectric.find(sid)->second;
        EXPECT_FALSE(di.is_metal);
        EXPECT_SOFT_EQ(1, specular_spike(di).y[0]);
        EXPECT_SOFT_EQ(0, specular_lobe(di).y[0]);
        EXPECT_SOFT_EQ(0, backscatter(di).y[0]);
    }

    // tube2_surf: glisur, ground, dielectric-dielectric, specular lobe
    {
        PhysSurfaceId sid{1};
        EXPECT_FALSE(OS_IS_MAPPED(roughness.polished, sid));
        EXPECT_TRUE(OS_IS_MAPPED(roughness.smear, sid));
        EXPECT_FALSE(OS_IS_MAPPED(roughness.gaussian, sid));

        EXPECT_TRUE(OS_IS_MAPPED(reflectivity.grid, sid));
        EXPECT_FALSE(OS_IS_MAPPED(reflectivity.fresnel, sid));

        EXPECT_TRUE(OS_IS_MAPPED(interaction.dielectric, sid));

        static double const polish
            = 1 - osp.roughness.smear.find(sid)->second.roughness;
        EXPECT_SOFT_EQ(0.9, polish);

        auto& di = osp.interaction.dielectric.find(sid)->second;
        EXPECT_FALSE(di.is_metal);
        EXPECT_SOFT_EQ(0, specular_spike(di).y[0]);
        EXPECT_SOFT_EQ(1, specular_lobe(di).y[0]);
        EXPECT_SOFT_EQ(0, backscatter(di).y[0]);
    }

    // lomid_surf: unified, polished, dielectric-dielectric
    {
        PhysSurfaceId sid{2};
        EXPECT_FALSE(OS_IS_MAPPED(roughness.polished, sid));
        EXPECT_FALSE(OS_IS_MAPPED(roughness.smear, sid));
        EXPECT_TRUE(OS_IS_MAPPED(roughness.gaussian, sid));

        EXPECT_TRUE(OS_IS_MAPPED(reflectivity.grid, sid));
        EXPECT_FALSE(OS_IS_MAPPED(reflectivity.fresnel, sid));

        EXPECT_TRUE(OS_IS_MAPPED(interaction.dielectric, sid));

        auto& di = osp.interaction.dielectric.find(sid)->second;
        EXPECT_FALSE(di.is_metal);
        static double const expected_energy[] = {2e-06, 8e-06};
        static double const expected_specular_spike[] = {0.1, 0.3};
        static double const expected_specular_lobe[] = {0.2, 0.2};
        static double const expected_backscatter[] = {0.3, 0.1};

        EXPECT_VEC_SOFT_EQ(expected_energy, specular_lobe(di).x);
        EXPECT_VEC_SOFT_EQ(expected_energy, specular_spike(di).x);
        EXPECT_VEC_SOFT_EQ(expected_energy, backscatter(di).x);
        EXPECT_VEC_SOFT_EQ(expected_specular_lobe, specular_lobe(di).y);
        EXPECT_VEC_SOFT_EQ(expected_specular_spike, specular_spike(di).y);
        EXPECT_VEC_SOFT_EQ(expected_backscatter, backscatter(di).y);
    }

    // midlo_surf: glisur, polished, dielectric-metal, specular spike
    {
        PhysSurfaceId sid{3};
        EXPECT_TRUE(OS_IS_MAPPED(roughness.polished, sid));
        EXPECT_FALSE(OS_IS_MAPPED(roughness.smear, sid));
        EXPECT_FALSE(OS_IS_MAPPED(roughness.gaussian, sid));

        EXPECT_TRUE(OS_IS_MAPPED(reflectivity.grid, sid));
        EXPECT_FALSE(OS_IS_MAPPED(reflectivity.fresnel, sid));

        EXPECT_TRUE(OS_IS_MAPPED(interaction.dielectric, sid));

        auto& di = osp.interaction.dielectric.find(sid)->second;
        EXPECT_TRUE(di.is_metal);
        EXPECT_SOFT_EQ(1, specular_spike(di).y[0]);
        EXPECT_SOFT_EQ(0, specular_lobe(di).y[0]);
        EXPECT_SOFT_EQ(0, backscatter(di).y[0]);
    }

    // midhi_surf: glisur, ground, dielectric-metal, specular lobe
    {
        PhysSurfaceId sid{4};
        EXPECT_FALSE(OS_IS_MAPPED(roughness.polished, sid));
        EXPECT_TRUE(OS_IS_MAPPED(roughness.smear, sid));
        EXPECT_FALSE(OS_IS_MAPPED(roughness.gaussian, sid));

        EXPECT_TRUE(OS_IS_MAPPED(reflectivity.grid, sid));
        EXPECT_FALSE(OS_IS_MAPPED(reflectivity.fresnel, sid));

        EXPECT_FALSE(OS_IS_MAPPED(interaction.dielectric, sid));

        static double const polish
            = 1 - osp.roughness.smear.find(sid)->second.roughness;
        EXPECT_SOFT_EQ(0.7, polish);

        auto& di = osp.interaction.dielectric.find(sid)->second;
        EXPECT_TRUE(di.is_metal);
        EXPECT_SOFT_EQ(0, specular_spike(di).y[0]);
        EXPECT_SOFT_EQ(1, specular_lobe(di).y[0]);
        EXPECT_SOFT_EQ(0, backscatter(di).y[0]);
    }

#undef OS_IS_MAPPED
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
