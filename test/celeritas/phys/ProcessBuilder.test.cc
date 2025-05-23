//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ProcessBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/ProcessBuilder.hh"

#include "corecel/sys/Environment.hh"
#include "celeritas/em/process/BremsstrahlungProcess.hh"
#include "celeritas/em/process/ComptonProcess.hh"
#include "celeritas/em/process/CoulombScatteringProcess.hh"
#include "celeritas/em/process/EIonizationProcess.hh"
#include "celeritas/em/process/EPlusAnnihilationProcess.hh"
#include "celeritas/em/process/GammaConversionProcess.hh"
#include "celeritas/em/process/MuBremsstrahlungProcess.hh"
#include "celeritas/em/process/MuIonizationProcess.hh"
#include "celeritas/em/process/PhotoelectricProcess.hh"
#include "celeritas/em/process/RayleighProcess.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/LivermorePEReader.hh"
#include "celeritas/io/NeutronXsReader.hh"
#include "celeritas/io/SeltzerBergerReader.hh"
#include "celeritas/neutron/process/NeutronElasticProcess.hh"
#include "celeritas/phys/Model.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<class T>
bool is_process_type(Process const* p)
{
    return dynamic_cast<T const*>(p) != nullptr;
}

#define EXPECT_PROCESS_TYPE(CLS, VALUE) \
    EXPECT_PRED1(is_process_type<CLS>, VALUE)

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ProcessBuilderTest : public Test
{
  protected:
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;

    using ActionIdIter = Process::ActionIdIter;
    using IPC = ImportProcessClass;

    static ImportData& import_data();
    static SPConstParticle& particle();
    static SPConstMaterial& material();

    static void SetUpTestCase()
    {
        ScopedRootErrorHandler scoped_root_error_;
        RootImporter import_from_root(
            Test::test_data_path("celeritas", "four-steel-slabs.root").c_str());
        import_data() = import_from_root();
        particle() = ParticleParams::from_import(import_data());
        material() = MaterialParams::from_import(import_data());
        CELER_ENSURE(particle() && material());
    }

    static bool has_env(std::string const& var)
    {
        bool result = !celeritas::getenv(var).empty();
        if (!result && strict_testing())
        {
            ADD_FAILURE() << "CI testing requires '" << var
                          << "' to be defined";
        }
        return result;
    }

    static bool has_le_data()
    {
        static bool const result = has_env("G4LEDATA");
        return result;
    }

    static bool has_neutron_data()
    {
        static bool const result = has_env("G4PARTICLEXSDATA");
        return result;
    }
};

ImportData& ProcessBuilderTest::import_data()
{
    static ImportData id;
    return id;
}

auto ProcessBuilderTest::particle() -> SPConstParticle&
{
    static SPConstParticle particle;
    return particle;
}

auto ProcessBuilderTest::material() -> SPConstMaterial&
{
    static SPConstMaterial material;
    return material;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ProcessBuilderTest, compton)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());
    // Create process
    auto process = build_process(IPC::compton);
    EXPECT_PROCESS_TYPE(ComptonProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("scat-klein-nishina", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            EXPECT_TRUE(process->macro_xs(applic));
            EXPECT_FALSE(process->energy_loss(applic));
        }

        // Test micro xs
        for (auto const& model : models)
        {
            EXPECT_TRUE(model->micro_xs(applic).empty());
        }
    }
}

TEST_F(ProcessBuilderTest, e_ionization)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());
    // Create process
    auto process = build_process(IPC::e_ioni);
    EXPECT_PROCESS_TYPE(EIonizationProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("ioni-moller-bhabha", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        for (auto applic : all_applic)
        {
            // Test step limits
            {
                applic.material = mat_id;
                EXPECT_TRUE(process->macro_xs(applic));
                EXPECT_TRUE(process->energy_loss(applic));
            }

            // Test micro xs
            for (auto const& model : models)
            {
                EXPECT_TRUE(model->micro_xs(applic).empty());
            }
        }
    }
}

TEST_F(ProcessBuilderTest, eplus_annihilation)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());
    // Create process
    auto process = build_process(IPC::annihilation);
    EXPECT_PROCESS_TYPE(EPlusAnnihilationProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("annihil-2-gamma", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        for (auto applic : all_applic)
        {
            // Test step limits
            {
                applic.material = mat_id;
                // Cross section calculated on the fly
                EXPECT_FALSE(process->macro_xs(applic));
                EXPECT_FALSE(process->energy_loss(applic));
            }

            // Test micro xs
            for (auto const& model : models)
            {
                EXPECT_TRUE(model->micro_xs(applic).empty());
            }
        }
    }
}

TEST_F(ProcessBuilderTest, gamma_conversion)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());
    // Create process
    auto process = build_process(IPC::conversion);
    EXPECT_PROCESS_TYPE(GammaConversionProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("conv-bethe-heitler", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            EXPECT_TRUE(process->macro_xs(applic));
            EXPECT_FALSE(process->energy_loss(applic));
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto micro_xs = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), micro_xs.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(micro_xs[elcomp_idx]);
            }
        }
    }
}

TEST_F(ProcessBuilderTest, photoelectric)
{
    if (!this->has_le_data())
    {
        GTEST_SKIP() << "Missing G4LEDATA";
    }

    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());
    // Create process
    auto process = build_process(IPC::photoelectric);
    EXPECT_PROCESS_TYPE(PhotoelectricProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("photoel-livermore", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            EXPECT_TRUE(process->macro_xs(applic));
            EXPECT_FALSE(process->energy_loss(applic));
        }

        // Test micro xs
        for (auto const& model : models)
        {
            EXPECT_TRUE(model->micro_xs(applic).empty());
        }
    }
}

TEST_F(ProcessBuilderTest, bremsstrahlung_multiple_models)
{
    if (!this->has_le_data())
    {
        GTEST_SKIP() << "Missing G4LEDATA";
    }

    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());

    // Create process
    auto process = build_process(IPC::e_brems);
    EXPECT_PROCESS_TYPE(BremsstrahlungProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(2, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("brems-sb", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());
    Applicability applic = *all_applic.begin();

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            EXPECT_TRUE(process->macro_xs(applic));

            // Only the ionization process has and energy loss table, which is
            // the sum of the ionization and bremsstrahlung energy loss
            EXPECT_FALSE(process->energy_loss(applic));
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto micro_xs = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), micro_xs.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(micro_xs[elcomp_idx]);
            }
        }
    }
}

TEST_F(ProcessBuilderTest, rayleigh)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());

    // Create process
    auto process = build_process(IPC::rayleigh);
    EXPECT_PROCESS_TYPE(RayleighProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("scat-rayleigh", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            EXPECT_TRUE(process->macro_xs(applic));
            EXPECT_FALSE(process->energy_loss(applic));
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto micro_xs = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), micro_xs.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(micro_xs[elcomp_idx]);
            }
        }
    }
}

TEST_F(ProcessBuilderTest, coulomb)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());

    // Create process
    auto process = build_process(IPC::coulomb_scat);
    EXPECT_PROCESS_TYPE(CoulombScatteringProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("coulomb-wentzel", models.front()->label());

    // Applicabilities for electron and positron
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());
    Applicability applic = *all_applic.begin();
    EXPECT_EQ(100, value_as<units::MevEnergy>(applic.lower));
    EXPECT_EQ(1e8, value_as<units::MevEnergy>(applic.upper));

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            EXPECT_TRUE(process->macro_xs(applic));
            EXPECT_FALSE(process->energy_loss(applic));
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto micro_xs = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), micro_xs.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(micro_xs[elcomp_idx]);
            }
        }
    }
}

TEST_F(ProcessBuilderTest, neutron_elastic)
{
    if (!this->has_neutron_data())
    {
        GTEST_SKIP() << "Missing G4PARTICLEXSDATA";
    }

    // Create ParticleParams with neutron
    ParticleParams::Input particle_inp = {
        {"neutron",
         pdg::neutron(),
         units::MevMass{939.5654133},
         zero_quantity(),
         constants::stable_decay_constant},
    };
    SPConstParticle particle_params
        = std::make_shared<ParticleParams>(std::move(particle_inp));

    ProcessBuilder build_process(
        this->import_data(), particle_params, this->material());

    // Create process
    auto process = build_process(IPC::neutron_elastic);
    EXPECT_PROCESS_TYPE(NeutronElasticProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("neutron-elastic-chips", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            // Cross section calculated on the fly
            EXPECT_FALSE(process->macro_xs(applic));
            EXPECT_FALSE(process->energy_loss(applic));
        }

        // Test micro xs
        for (auto const& model : models)
        {
            EXPECT_TRUE(model->micro_xs(applic).empty());
        }
    }
}

TEST_F(ProcessBuilderTest, mu_ionization)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());

    // Create process
    auto process = build_process(IPC::mu_ioni);
    EXPECT_PROCESS_TYPE(MuIonizationProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    // Note that newer versions of Geant4 use the \c G4MuBetheBloch model for
    // all energies above 200 so will only have three models
    ASSERT_EQ(3, models.size());
    ASSERT_TRUE(models[0]);
    EXPECT_EQ("ioni-icru73qo", models[0]->label());
    EXPECT_EQ(1, models[0]->applicability().size());
    ASSERT_TRUE(models[1]);
    EXPECT_EQ("ioni-bragg", models[1]->label());
    EXPECT_EQ(1, models[1]->applicability().size());
    ASSERT_TRUE(models[2]);
    EXPECT_EQ("ioni-mu-bethe-bloch", models[2]->label());
    auto all_applic = models[2]->applicability();
    ASSERT_EQ(2, all_applic.size());

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        for (auto applic : all_applic)
        {
            // Test step limits
            {
                applic.material = mat_id;
                EXPECT_TRUE(process->macro_xs(applic));
                EXPECT_TRUE(process->energy_loss(applic));
            }

            // Test micro xs
            for (auto const& model : models)
            {
                EXPECT_TRUE(model->micro_xs(applic).empty());
            }
        }
    }
}

TEST_F(ProcessBuilderTest, mu_bremsstrahlung)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material());

    // Create process
    auto process = build_process(IPC::mu_brems);
    EXPECT_PROCESS_TYPE(MuBremsstrahlungProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("brems-muon", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());
    Applicability applic = *all_applic.begin();

    for (auto mat_id : range(PhysMatId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            EXPECT_TRUE(process->macro_xs(applic));
            // Only the ionization process has and energy loss table, which is
            // the sum of the ionization and bremsstrahlung energy loss
            EXPECT_FALSE(process->energy_loss(applic));
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto micro_xs = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), micro_xs.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(micro_xs[elcomp_idx]);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
