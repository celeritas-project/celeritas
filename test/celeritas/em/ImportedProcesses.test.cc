//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/ImportedProcesses.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/process/BremsstrahlungProcess.hh"
#include "celeritas/em/process/ComptonProcess.hh"
#include "celeritas/em/process/EIonizationProcess.hh"
#include "celeritas/em/process/EPlusAnnihilationProcess.hh"
#include "celeritas/em/process/GammaConversionProcess.hh"
#include "celeritas/em/process/MultipleScatteringProcess.hh"
#include "celeritas/em/process/PhotoelectricProcess.hh"
#include "celeritas/em/process/RayleighProcess.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"

#include "celeritas_test.hh"

using namespace celeritas;
using VGT = ValueGridType;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ImportedProcessesTest : public celeritas_test::Test
{
  protected:
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstMaterials = std::shared_ptr<const MaterialParams>;
    using SPConstImported  = std::shared_ptr<const ImportedProcesses>;
    using ActionIdIter     = Process::ActionIdIter;

    void SetUp() override
    {
        RootImporter import_from_root(
            this->test_data_path("celeritas", "four-steel-slabs.root").c_str());

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
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("scat-klein-nishina", models.front()->label());
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
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("ioni-moller-bhabha", models.front()->label());
    auto all_applic = models.front()->applicability();
    EXPECT_EQ(2, all_applic.size());

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
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("annihil-2-gamma", models.front()->label());
    auto all_applic = models.front()->applicability();
    EXPECT_EQ(1, all_applic.size());

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
    GammaConversionProcess::Options options;
    options.enable_lpm = true;

    // Create gamma conversion process
    auto process = std::make_shared<GammaConversionProcess>(
        particles_, processes_, options);

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("conv-bethe-heitler", models.front()->label());
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

TEST_F(ImportedProcessesTest, msc)
{
    // Create Multiple scattering process
    auto process = std::make_shared<MultipleScatteringProcess>(
        particles_, materials_, processes_);

    // Test model
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("msc-urban", models.front()->label());
    auto all_applic = models.front()->applicability();
    ASSERT_EQ(2, all_applic.size());
    Applicability applic = *all_applic.begin();

    // Test step limits
    for (auto mat_id : range(MaterialId{materials_->num_materials()}))
    {
        applic.material = mat_id;
        auto builders   = process->step_limits(applic);
        EXPECT_TRUE(builders[VGT::msc_mfp]);
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
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("photoel-livermore", models.front()->label());
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
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(2, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("brems-sb", models.front()->label());
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
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("brems-combined", models.front()->label());
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
    auto models = process->build_models(ActionIdIter{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("scat-rayleigh", models.front()->label());
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
