//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Rayleigh.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/Histogram.hh"
#include "corecel/random/HistogramSampler.hh"
#include "celeritas/optical/interactor/RayleighInteractor.hh"
#include "celeritas/optical/model/RayleighMfpCalculator.hh"
#include "celeritas/optical/model/RayleighModel.hh"

#include "InteractorHostTestBase.hh"
#include "OpticalMockTestBase.hh"
#include "TestMacros.hh"
#include "ValidationUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RayleighInteractorTest : public InteractorHostTestBase
{
  protected:
    void SetUp() override
    {
        // Check incident quantities are valid
        this->check_direction_polarization(
            this->direction(), this->particle_track().polarization());
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Interactions should always be scattering
        EXPECT_EQ(Interaction::Action::scattered, interaction.action);
        this->check_direction_polarization(interaction);
    }
};

class RayleighModelTest : public OpticalMockTestBase
{
  protected:
    void SetUp() override {}

    //! Create Rayleigh model from mock data
    std::shared_ptr<RayleighModel const> create_model()
    {
        auto models = std::make_shared<ImportedModels const>(
            this->imported_data().optical_models);
        return std::make_shared<RayleighModel const>(
            ActionId{0}, models, RayleighModel::Input{});
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Basic tests for Rayleigh scattering interaction
TEST_F(RayleighInteractorTest, TEST_IF_CELERITAS_DOUBLE(basic))
{
    int const num_samples = 4;

    RayleighInteractor interact{this->particle_track(), this->direction()};

    auto& rng_engine = this->rng();

    std::vector<real_type> dir_angle;
    std::vector<real_type> pol_angle;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        this->sanity_check(result);

        dir_angle.push_back(dot_product(result.direction, this->direction()));
        pol_angle.push_back(dot_product(
            result.polarization, this->particle_track().polarization()));
    }

    static real_type const expected_dir_angle[] = {
        -0.72904599140644,
        0.99292265109602,
        -0.78027649831159,
        -0.77507096788764,
    };
    static real_type const expected_pol_angle[] = {
        -0.93732637186049,
        -0.99321124082734,
        0.98251616641497,
        -0.9149817471032,
    };

    EXPECT_EQ(40, rng_engine.count());
    EXPECT_VEC_SOFT_EQ(expected_dir_angle, dir_angle);
    EXPECT_VEC_SOFT_EQ(expected_pol_angle, pol_angle);
}

//---------------------------------------------------------------------------//
// Test statistical consistency over larger number of samples
TEST_F(RayleighInteractorTest, stress_test)
{
    constexpr size_type num_samples = 1'000'000;

    RayleighInteractor interact{this->particle_track(), this->direction()};

    auto& rng_engine = this->rng();
    auto const inc_dir = this->direction();
    auto const inc_pol = this->particle_track().polarization();

    Histogram accum_dir{8, {-1, 1}};
    Histogram accum_pol{8, {-1, 1}};
    accumulate_n(
        [&](auto&& result) {
            accum_dir(dot_product(result.direction, inc_dir));
            accum_pol(dot_product(result.polarization, inc_pol));
        },
        interact,
        rng_engine,
        num_samples);
    EXPECT_FALSE(accum_dir.underflow() || accum_dir.overflow()
                 || accum_pol.underflow() || accum_pol.overflow());

    static double const expected_accum_dir[] = {
        0.664064,
        0.523436,
        0.431044,
        0.38252,
        0.383708,
        0.42894,
        0.523428,
        0.66286,
    };
    static double const expected_accum_pol[] = {
        1.696864,
        0.25238,
        0.04754,
        0.003144,
        0.002992,
        0.04892,
        0.252776,
        1.695384,
    };

    auto avg_samples = static_cast<double>(rng_engine.exchange_count())
                       / static_cast<double>(num_samples);
    real_type tol{1e-2};
    EXPECT_VEC_CLOSE(expected_accum_dir, accum_dir.calc_density(), tol, tol);
    EXPECT_VEC_CLOSE(expected_accum_pol, accum_pol.calc_density(), tol, tol);
    EXPECT_SOFT_NEAR(
        6.0016 * sizeof(real_type) / sizeof(float), avg_samples, tol);
}

//---------------------------------------------------------------------------//
// Check model name and description are properly initialized
TEST_F(RayleighModelTest, description)
{
    auto model = create_model();

    EXPECT_EQ(ActionId{0}, model->action_id());
    EXPECT_EQ("optical-rayleigh", model->label());
    EXPECT_EQ("interact by optical Rayleigh", model->description());
}

//---------------------------------------------------------------------------//
// Check Rayleigh model MFP tables match imported ones
TEST_F(RayleighModelTest, interaction_mfp)
{
    OwningGridAccessor storage;

    auto model = create_model();
    auto builder = storage.create_mfp_builder();

    for (auto mat : range(OptMatId(this->num_optical_materials())))
    {
        model->build_mfps(mat, builder);
    }

    EXPECT_TABLE_EQ(
        this->import_model_by_class(ImportModelClass::rayleigh).mfp_table,
        storage(builder.grid_ids()));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
