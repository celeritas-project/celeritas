//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Mie.test.cc
//---------------------------------------------------------------------------//
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

#include "InteractorHostTestBase.hh"
#include "OpticalMockTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MieTest : public InteractorHostBase, public OpticalMockTestBase
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

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MieTest, mie_params)
{
    this->build_model();

    // Test the material properties of mie scattering parameters
    MieMaterialData mie_record = data_.mie_record[material_id_];
    EXPECT_SOFT_EQ(0.99, mie_record.forward_g);
    EXPECT_SOFT_EQ(0.99, mie_record.backward_g);
    EXPECT_SOFT_EQ(0.80, mie_record.forward_ratio);
}

TEST_F(MieTest, mie_basic)
{
    int const num_samples = 4;

    this->build_model();

    MieInteractor interact(
        data_, this->particle_track(), direction_, material_id_);

    auto& rng_engine = this->InteractorHostBase::rng();
    this->set_inc_polarization({0, 1, 0});

    std::vector<real_type> dir_angle;
    std::vector<real_type> pol_angle;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        this->check_direction_polarization(result);

        // Store dot products with incident direction/polarization
        dir_angle.push_back(dot_product(result.direction, this->direction()));
        pol_angle.push_back(dot_product(
            result.polarization, this->particle_track().polarization()));
    }

    static real_type const expected_dir_angle[] = {
        0.997467127484242,
        0.999530487034177,
        0.999999642467185,
        0.996187032055894,
    };
    static real_type const expected_pol_angle[] = {
        0.999904430863429,
        -0.99959742953257,
        -0.999999650643697,
        0.996510957439599,
    };

    EXPECT_EQ(32, rng_engine.count());
    EXPECT_VEC_SOFT_EQ(expected_dir_angle, dir_angle);
    EXPECT_VEC_SOFT_EQ(expected_pol_angle, pol_angle);
}

TEST_F(MieTest, mfp)
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
