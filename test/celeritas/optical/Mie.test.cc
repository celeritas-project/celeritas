//---------------------------------*- C++
//-*----------------------------------//
// Copyright ...
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Mie.test.cc
//! \brief Unit tests for Mie optical scattering (model, interactor, executor).
//---------------------------------------------------------------------------//

#include <memory>

#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/MieData.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/ValidationUtils.hh"
#include "celeritas/optical/interactor/MieInteractor.hh"
#include "celeritas/optical/model/MieExecutor.hh"
#include "celeritas/optical/model/MieModel.hh"
// #include "celeritas/phys/InteractorHostTestBase.hh"

#include "InteractorHostTestBase.hh"
#include "OpticalMockTestBase.hh"
#include "Test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
// MieModelTest
//---------------------------------------------------------------------------//

class MieModelTest : public InteractorHostBase, public OpticalMockTestBase
{
  protected:
    using HostDataCRef = HostCRef<MieData>;
    void SetUp() override {}
    void build_model()
    {
        auto const& data = this->imported_data();
        MieModel::Input input;
        input.model = ImportModelClass::mie;
        for (auto const& mat : data.optical_materials)
        {
            input.data.push_back(mat.mie);
        }
        auto models
            = std::make_shared<ImportedModels const>(data.optical_models);
        model_ = std::make_shared<MieModel const>(ActionId{0}, models, input);
        data_ = model_->host_ref();
    }

    OptMatId material_id_{0};
    Real3 direction_{0, 0, 1};
    std::shared_ptr<MieModel const> model_;
    HostCRef<MieData> data_;
};

TEST_F(MieModelTest, mie_basic)
{
    this->build_model();
    //  int const num_samples = 4;
    // also perhaps look at what this action is they are mentioning about if it
    // needs to be scattered or what
    auto& rng = this->InteractorHostBase::rng();
    real_type test_energy = 2e-6;
    this->set_inc_energy(Energy{test_energy});

    MieInteractor interact(
        data_, this->particle_track(), direction_, material_id_);

    auto result = interact(rng);

    this->check_direction_polarization(result);
    EXPECT_TRUE(is_soft_unit_vector(result.direction));
    EXPECT_TRUE(is_soft_unit_vector(result.polarization));
    auto const& host_ref = model_->host_ref();
    ASSERT_FALSE(host_ref.mie_record.empty());
    EXPECT_GT(host_ref.mie_record.size(), 0);
}

TEST_F(MieModelTest, mfp)
{
    OwningGridAccessor storage;

    this->build_model();
    auto builder = storage.create_mfp_builder();

    for (auto mat : range(OptMatId(this->num_optical_materials())))
    {
        model_->build_mfps(mat, builder);
    }

    EXPECT_TABLE_EQ(
        this->import_model_by_class(ImportModelClass::mie).mfp_table,
        storage(builder.grid_ids()));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
