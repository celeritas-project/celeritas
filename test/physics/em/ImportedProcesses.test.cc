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
#include "io/LivermorePEParamsReader.hh"
#include "io/RootImporter.hh"
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

        auto data  = import_from_root();
        particles_ = std::move(data.particle_params);
        materials_ = std::move(data.material_params);
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

TEST_F(ImportedProcessesTest, livermore)
{
    if (!CELERITAS_USE_CUDA)
    {
        // constructor contains device_pointers because it doesn't use
        // Collection
        SKIP("FIXME: livermore model currently requires CUDA");
    }

    // Set up livermore params reader (requires Geant4 environment variables)
    std::unique_ptr<LivermorePEParamsReader> read_el;
    try
    {
        read_el = std::make_unique<LivermorePEParamsReader>();
    }
    catch (const RuntimeError& e)
    {
        SKIP("Failed to set up reader: " << e.what());
    }

    // Load livermore data
    LivermorePEParams::Input li;
    for (auto el_id : range(ElementId{materials_->num_elements()}))
    {
        auto el_view = materials_->get(el_id);
        li.elements.push_back((*read_el)(el_view.atomic_number()));
    }
    auto livermore_data = std::make_shared<LivermorePEParams>(std::move(li));

    // Create photoelectric process
    auto process = std::make_shared<PhotoelectricProcess>(
        particles_, processes_, livermore_data);

    // Test model
    auto models = process->build_models(ModelIdGenerator{});
    ASSERT_EQ(1, models.size());
    ASSERT_TRUE(models.front());
    EXPECT_EQ("Livermore", models.front()->label());
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
