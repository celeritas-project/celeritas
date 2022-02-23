//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportedProcesses.test.cc
//---------------------------------------------------------------------------//
#include "io/ImportData.hh"
#include "io/RootImporter.hh"
#include "physics/base/ImportedProcessAdapter.hh"
#include "physics/base/Model.hh"
#include "physics/em/BremsstrahlungProcess.hh"
#include "physics/em/ComptonProcess.hh"
#include "physics/em/EIonizationProcess.hh"
#include "physics/em/EPlusAnnihilationProcess.hh"
#include "physics/em/GammaConversionProcess.hh"
#include "physics/em/PhotoelectricProcess.hh"
#include "physics/em/RayleighProcess.hh"

#include "celeritas_test.hh"

using namespace celeritas;
using VGT = ValueGridType;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ImportedProcessesTest : public celeritas::Test
{
  protected:
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstMaterials = std::shared_ptr<const MaterialParams>;
    using SPConstImported  = std::shared_ptr<const ImportedProcesses>;

    void SetUp() override
    {
        RootImporter import_from_root(
            this->test_data_path("io", "geant-exporter-data.root").c_str());

        auto data = import_from_root();

        particles_ = ParticleParams::from_import(data);
        materials_ = MaterialParams::from_import(data);
        processes_
            = std::make_shared<ImportedProcesses>(std::move(data.processes));

        CELER_ENSURE(particles_);
        CELER_ENSURE(processes_->size() > 0);
    }

    SPConstParticles particles_;
    SPConstMaterials materials_;
    SPConstImported  processes_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ImportedProcessesTest, compton)
{
    // Create Compton process
    auto process = std::make_shared<ComptonProcess>(particles_, processes_);

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Klein-Nishina", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        applic.material = mat_id;
        auto builders   = process->step_limits(applic);
        EXPECT_TRUE(builders[VGT::macro_xs]);
        EXPECT_FALSE(builders[VGT::energy_loss]);
        EXPECT_FALSE(builders[VGT::range]);
    }
}

TEST_F(ImportedProcessesTest, e_ionization)
{
    // Create electron ionization process
    auto process = std::make_shared<EIonizationProcess>(particles_, processes_);

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Moller/Bhabha scattering", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        for (auto applic : all_applic)
        {
            applic.material = mat_id;
            auto builders   = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);
            EXPECT_TRUE(builders[VGT::energy_loss]);
            EXPECT_TRUE(builders[VGT::range]);
        }
    }
}

TEST_F(ImportedProcessesTest, eplus_annihilation)
{
    // Create positron annihilation process
    auto process = std::make_shared<EPlusAnnihilationProcess>(particles_);

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Positron annihilation (2g)", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        for (auto applic : all_applic)
        {
            applic.material = mat_id;
            auto builders   = process->step_limits(applic);
            EXPECT_TRUE(builders[VGT::macro_xs]);
            EXPECT_FALSE(builders[VGT::energy_loss]);
            EXPECT_FALSE(builders[VGT::range]);
        }
    }
}

TEST_F(ImportedProcessesTest, gamma_conversion)
{
    // Create gamma conversion process
    auto process
        = std::make_shared<GammaConversionProcess>(particles_, processes_);

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Bethe-Heitler", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        applic.material = mat_id;
        auto builders   = process->step_limits(applic);
        EXPECT_TRUE(builders[VGT::macro_xs]);
        EXPECT_FALSE(builders[VGT::energy_loss]);
        EXPECT_FALSE(builders[VGT::range]);
    }
}

TEST_F(ImportedProcessesTest, photoelectric)
{
    // Create photoelectric process (requires Geant4 environment variables)
    std::shared_ptr<PhotoelectricProcess> process;
    try
    {
        process = std::make_shared<PhotoelectricProcess>(
            particles_, materials_, processes_);
    }
    catch (const RuntimeError& e)
    {
        GTEST_SKIP() << "Failed to create process: " << e.what();
    }

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Livermore photoelectric", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        applic.material = mat_id;
        auto builders   = process->step_limits(applic);
        EXPECT_TRUE(builders[VGT::macro_xs]);
        EXPECT_FALSE(builders[VGT::energy_loss]);
        EXPECT_FALSE(builders[VGT::range]);
    }
}

TEST_F(ImportedProcessesTest, bremsstrahlung_multiple_models)
{
    // Create bremsstrahlung process (requires Geant4 environment variables)
    // with multiple models (SeltzerBergerModel and RelativisticBremModel)
    BremsstrahlungProcess::Options options;
    options.combined_model = false;
    std::shared_ptr<BremsstrahlungProcess> process;
    try
    {
        process = std::make_shared<BremsstrahlungProcess>(
            particles_, materials_, processes_, options);
    }
    catch (const RuntimeError& e)
    {
        GTEST_SKIP() << "Failed to create process: " << e.what();
    }

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(2, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Seltzer-Berger bremsstrahlung", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());
    Applicability applic = *all_applic.begin();

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        applic.material = mat_id;
        auto builders   = process->step_limits(applic);
        EXPECT_TRUE(builders[VGT::macro_xs]);

        // Only the ionization process has energy loss and range tables. It's
        // de/dx table is the sum of the ionization and bremsstrahlung energy
        // loss, and the range table is calculated from the summed de/dx.
        EXPECT_FALSE(builders[VGT::energy_loss]);
        EXPECT_FALSE(builders[VGT::range]);
    }
}

TEST_F(ImportedProcessesTest, bremsstrahlung_combined_model)
{
    // Create the combined bremsstrahlung process
    BremsstrahlungProcess::Options options;
    options.combined_model = true;
    std::shared_ptr<BremsstrahlungProcess> process;
    try
    {
        process = std::make_shared<BremsstrahlungProcess>(
            particles_, materials_, processes_, options);
    }
    catch (const RuntimeError& e)
    {
        GTEST_SKIP() << "Failed to create process: " << e.what();
    }

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Combined Bremsstrahlung", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());
    Applicability applic = *all_applic.begin();

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        applic.material = mat_id;
        auto builders   = process->step_limits(applic);
        EXPECT_TRUE(builders[VGT::macro_xs]);

        // Only the ionization process has energy loss and range tables. It's
        // de/dx table is the sum of the ionization and bremsstrahlung energy
        // loss, and the range table is calculated from the summed de/dx.
        EXPECT_FALSE(builders[VGT::energy_loss]);
        EXPECT_FALSE(builders[VGT::range]);
    }
}

TEST_F(ImportedProcessesTest, rayleigh)
{
    // Create Rayleigh scattering process
    auto process = std::make_shared<RayleighProcess>(
        particles_, materials_, processes_);

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Rayleigh Scattering", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(1, all_applic.size());
    Applicability applic = *all_applic.begin();

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        applic.material = mat_id;
        auto builders   = process->step_limits(applic);
        EXPECT_TRUE(builders[VGT::macro_xs]);
        EXPECT_FALSE(builders[VGT::energy_loss]);
        EXPECT_FALSE(builders[VGT::range]);
    }
}
