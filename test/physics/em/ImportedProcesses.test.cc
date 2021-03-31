//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportedProcesses.test.cc
//---------------------------------------------------------------------------//
#include "physics/base/ImportedProcessAdapter.hh"

#include "physics/base/Model.hh"
#include "physics/em/ComptonProcess.hh"
#include "physics/em/PhotoelectricProcess.hh"
#include "physics/em/EIonizationProcess.hh"
#include "io/LivermorePEReader.hh"
#include "io/RootLoader.hh"
#include "io/MaterialParamsLoader.hh"
#include "io/ParticleParamsLoader.hh"
#include "io/ImportProcessLoader.hh"

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
        std::string root_file
            = this->test_data_path("io", "geant-exporter-data.root");
        RootLoader root_loader(root_file.c_str());

        particles_ = std::move(ParticleParamsLoader(root_loader)());
        materials_ = std::move(MaterialParamsLoader(root_loader)());
        processes_ = std::make_shared<ImportedProcesses>(
            std::move(ImportProcessLoader(root_loader)()));

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
    // Create photoelectric process
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

TEST_F(ImportedProcessesTest, eionization)
{
    // Create photoelectric process
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
        SKIP("Failed to create process: " << e.what());
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
