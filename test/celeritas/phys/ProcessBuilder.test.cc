//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "celeritas/em/process/PhotoelectricProcess.hh"
#include "celeritas/em/process/RayleighProcess.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/LivermorePEReader.hh"
#include "celeritas/io/SeltzerBergerReader.hh"
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
    using Options = ProcessBuilder::Options;
    using VGT = ValueGridType;
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

    static bool has_le_data()
    {
        static bool const result = !celeritas::getenv("G4LEDATA").empty();
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
        this->import_data(), this->particle(), this->material(), Options{});
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

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            auto builders = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);
            EXPECT_FALSE(builders[VGT::energy_loss]);
            EXPECT_FALSE(builders[VGT::range]);
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto builders = model->micro_xs(applic);
            EXPECT_TRUE(builders.empty());
        }
    }
}

TEST_F(ProcessBuilderTest, e_ionization)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material(), Options{});
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

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        for (auto applic : all_applic)
        {
            // Test step limits
            {
                applic.material = mat_id;
                auto builders = process->step_limits(applic);
                EXPECT_TRUE(builders[VGT::macro_xs]);
                EXPECT_TRUE(builders[VGT::energy_loss]);
                EXPECT_TRUE(builders[VGT::range]);
            }

            // Test micro xs
            for (auto const& model : models)
            {
                auto builders = model->micro_xs(applic);
                EXPECT_TRUE(builders.empty());
            }
        }
    }
}

TEST_F(ProcessBuilderTest, eplus_annihilation)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material(), Options{});
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

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        for (auto applic : all_applic)
        {
            // Test step limits
            {
                applic.material = mat_id;
                auto builders = process->step_limits(applic);
                EXPECT_TRUE(builders[VGT::macro_xs]);
                EXPECT_FALSE(builders[VGT::energy_loss]);
                EXPECT_FALSE(builders[VGT::range]);
            }

            // Test micro xs
            for (auto const& model : models)
            {
                auto builders = model->micro_xs(applic);
                EXPECT_TRUE(builders.empty());
            }
        }
    }
}

TEST_F(ProcessBuilderTest, gamma_conversion)
{
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material(), Options{});
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

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            auto builders = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);
            EXPECT_FALSE(builders[VGT::energy_loss]);
            EXPECT_FALSE(builders[VGT::range]);
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto builders = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), builders.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(builders[elcomp_idx]);
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
        this->import_data(), this->particle(), this->material(), Options{});
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

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            auto builders = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);
            EXPECT_FALSE(builders[VGT::energy_loss]);
            EXPECT_FALSE(builders[VGT::range]);
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto builders = model->micro_xs(applic);
            EXPECT_TRUE(builders.empty());
        }
    }
}

TEST_F(ProcessBuilderTest, bremsstrahlung_multiple_models)
{
    if (!this->has_le_data())
    {
        GTEST_SKIP() << "Missing G4LEDATA";
    }

    Options pbopts;
    pbopts.brem_combined = false;
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material(), pbopts);

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

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            auto builders = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);

            // Only the ionization process has energy loss and range tables.
            // It's de/dx table is the sum of the ionization and bremsstrahlung
            // energy loss, and the range table is calculated from the summed
            // de/dx.
            EXPECT_FALSE(builders[VGT::energy_loss]);
            EXPECT_FALSE(builders[VGT::range]);
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto builders = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), builders.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(builders[elcomp_idx]);
            }
        }
    }
}

TEST_F(ProcessBuilderTest, bremsstrahlung_combined_model)
{
    if (!this->has_le_data())
    {
        GTEST_SKIP() << "Missing G4LEDATA";
    }

    Options pbopts;
    pbopts.brem_combined = true;
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material(), pbopts);

    // Create process
    auto process = build_process(IPC::e_brems);
    EXPECT_PROCESS_TYPE(BremsstrahlungProcess, process.get());

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("brems-combined", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());
    Applicability applic = *all_applic.begin();

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            auto builders = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);

            // Only the ionization process has energy loss and range tables.
            // It's de/dx table is the sum of the ionization and bremsstrahlung
            // energy loss, and the range table is calculated from the summed
            // de/dx.
            EXPECT_FALSE(builders[VGT::energy_loss]);
            EXPECT_FALSE(builders[VGT::range]);
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto builders = model->micro_xs(applic);
            EXPECT_TRUE(builders.empty());
        }
    }
}

TEST_F(ProcessBuilderTest, rayleigh)
{
    Options pbopts;
    pbopts.brem_combined = false;
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material(), pbopts);

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

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            auto builders = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);
            EXPECT_FALSE(builders[VGT::energy_loss]);
            EXPECT_FALSE(builders[VGT::range]);
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto builders = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), builders.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(builders[elcomp_idx]);
            }
        }
    }
}

TEST_F(ProcessBuilderTest, coulomb)
{
    Options pbopts;
    pbopts.brem_combined = false;
    ProcessBuilder build_process(
        this->import_data(), this->particle(), this->material(), pbopts);

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

    for (auto mat_id : range(MaterialId{this->material()->num_materials()}))
    {
        // Test step limits
        {
            applic.material = mat_id;
            auto builders = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);
            EXPECT_FALSE(builders[VGT::energy_loss]);
            EXPECT_FALSE(builders[VGT::range]);
        }

        // Test micro xs
        for (auto const& model : models)
        {
            auto builders = model->micro_xs(applic);
            auto material = this->material()->get(mat_id);
            EXPECT_EQ(material.num_elements(), builders.size());
            for (auto elcomp_idx : range(material.num_elements()))
            {
                EXPECT_TRUE(builders[elcomp_idx]);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
