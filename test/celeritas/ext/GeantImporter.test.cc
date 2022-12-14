//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantImporter.hh"

#include "corecel/io/Repr.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/io/ImportData.hh"

#include "celeritas_test.hh"

using namespace celeritas;

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
} // namespace

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeantImporterTest : public celeritas_test::Test
{
  protected:
    using DataSelection = GeantImporter::DataSelection;

    struct ImportSummary
    {
        std::vector<std::string> particles;
        std::vector<std::string> processes;
        std::vector<std::string> models;
        void                     print_expected() const;
    };

    ImportData import_geant(const DataSelection& selection)
    {
        // Only allow a single importer per global execution, because of Geant4
        // limitations.
        static GeantImporter import_geant(this->setup_geant());

        return import_geant(selection);
    }

    ImportSummary summarize(const ImportData& data) const;

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
    using celeritas::repr;
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

//---------------------------------------------------------------------------//
class FourSteelSlabsEmStandard : public GeantImporterTest
{
    GeantSetup setup_geant() override
    {
        GeantSetup::Options opts;
        opts.physics = GeantSetupPhysicsList::em_standard;
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

    static const char* expected_particles[]
        = {"e+", "e-", "gamma", "mu+", "mu-"};
    EXPECT_VEC_EQ(expected_particles, summary.particles);
    static const char* expected_processes[] = {"msc",
                                               "coulomb_scat",
                                               "e_ioni",
                                               "e_brems",
                                               "photoelectric",
                                               "compton",
                                               "conversion",
                                               "rayleigh",
                                               "mu_ioni",
                                               "mu_brems",
                                               "mu_pair_prod"};
    EXPECT_VEC_EQ(expected_processes, summary.processes);
    static const char* expected_models[] = {"bethe_bloch",
                                            "urban_msc",
                                            "icru_73_qo",
                                            "wentzel_VI_uni",
                                            "e_coulomb_scattering",
                                            "bragg",
                                            "moller_bhabha",
                                            "e_brems_sb",
                                            "e_brems_lpm",
                                            "livermore_photoelectric",
                                            "klein_nishina",
                                            "bethe_heitler_lpm",
                                            "livermore_rayleigh",
                                            "mu_bethe_bloch",
                                            "mu_brems",
                                            "mu_pair_prod"};
    EXPECT_VEC_EQ(expected_models, summary.models);
}

TEST_F(FourSteelSlabsEmStandard, em_hadronic)
{
    DataSelection options;
    options.particles = DataSelection::em | DataSelection::hadron;
    options.processes = DataSelection::em;

    auto imported = this->import_geant(options);
    auto summary  = this->summarize(imported);

    static const char* expected_particles[] = {"He3",
                                               "alpha",
                                               "anti_He3",
                                               "anti_alpha",
                                               "anti_deuteron",
                                               "anti_doublehyperH4",
                                               "anti_doublehyperdoubleneutron",
                                               "anti_hyperH4",
                                               "anti_hyperHe5",
                                               "anti_hyperalpha",
                                               "anti_hypertriton",
                                               "anti_proton",
                                               "anti_triton",
                                               "deuteron",
                                               "doublehyperH4",
                                               "doublehyperdoubleneutron",
                                               "e+",
                                               "e-",
                                               "gamma",
                                               "hyperH4",
                                               "hyperHe5",
                                               "hyperalpha",
                                               "hypertriton",
                                               "kaon+",
                                               "kaon-",
                                               "mu+",
                                               "mu-",
                                               "pi+",
                                               "pi-",
                                               "proton",
                                               "triton"};
    EXPECT_VEC_EQ(expected_particles, summary.particles);
    static const char* expected_processes[] = {"ion_ioni",
                                               "msc",
                                               "h_ioni",
                                               "h_brems",
                                               "h_pair_prod",
                                               "coulomb_scat",
                                               "e_ioni",
                                               "e_brems",
                                               "photoelectric",
                                               "compton",
                                               "conversion",
                                               "rayleigh",
                                               "mu_ioni",
                                               "mu_brems",
                                               "mu_pair_prod"};
    EXPECT_VEC_EQ(expected_processes, summary.processes);
    static const char* expected_models[] = {"bragg_ion",
                                            "bethe_bloch",
                                            "urban_msc",
                                            "icru_73_qo",
                                            "wentzel_VI_uni",
                                            "h_brems",
                                            "h_pair_prod",
                                            "e_coulomb_scattering",
                                            "bragg",
                                            "moller_bhabha",
                                            "e_brems_sb",
                                            "e_brems_lpm",
                                            "livermore_photoelectric",
                                            "klein_nishina",
                                            "bethe_heitler_lpm",
                                            "livermore_rayleigh",
                                            "mu_bethe_bloch",
                                            "mu_brems",
                                            "mu_pair_prod"};
    EXPECT_VEC_EQ(expected_models, summary.models);
}
