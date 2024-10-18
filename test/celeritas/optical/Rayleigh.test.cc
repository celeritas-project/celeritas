//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalRayleigh.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/interactor/RayleighInteractor.hh"
#include "celeritas/optical/model/RayleighModel.hh"

#include "InteractorHostTestBase.hh"
#include "MockImportedData.hh"
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

class RayleighModelTest : public MockImportedData
{
  protected:
    void SetUp() override {}

    //! Construct Rayleigh model from mock data
    std::shared_ptr<RayleighModel const> create_model()
    {
        auto models = MockImportedData::create_imported_models();

        import_model_id_ = models->builtin_model_id(ImportModelClass::rayleigh);

        return std::make_shared<RayleighModel const>(
            ActionId{0},
            ImportedModelAdapter{ImportModelClass::rayleigh, models},
            this->create_input());
    }

    //! Construct Rayleigh model with only mock material data
    std::shared_ptr<RayleighModel const> create_empty_model()
    {
        auto models = MockImportedData::create_empty_imported_models();

        import_model_id_ = models->builtin_model_id(ImportModelClass::rayleigh);

        return std::make_shared<RayleighModel const>(
            ActionId{0},
            ImportedModelAdapter{ImportModelClass::rayleigh, models},
            this->create_input());
    }

    //! Create mock input for Rayleigh model
    RayleighModel::Input create_input() const
    {
        RayleighModel::Input input;
        for (auto const& material : this->import_materials())
        {
            input.properties.push_back(material.properties);
            input.rayleigh.push_back(material.rayleigh);
        }
        return input;
    }

    ImportedModels::ImportedModelId import_model_id_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Basic tests for Rayleigh scattering interaction
TEST_F(RayleighInteractorTest, basic)
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
    int const num_samples = 1'000;

    RayleighInteractor interact{this->particle_track(), this->direction()};

    auto& rng_engine = this->rng();

    Array<real_type, 2> dir_moment = {0, 0};
    Array<real_type, 2> pol_moment = {0, 0};

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        this->sanity_check(result);

        real_type dir_angle = dot_product(result.direction, this->direction());
        dir_moment[0] += dir_angle;
        dir_moment[1] += ipow<2>(dir_angle);

        real_type pol_angle = dot_product(
            result.polarization, this->particle_track().polarization());
        pol_moment[0] += pol_angle;
        pol_moment[1] += ipow<2>(pol_angle);
    }

    dir_moment /= num_samples;
    pol_moment /= num_samples;

    static real_type const expected_dir_moment[]
        = {-0.016961118324494, 0.39183211598064};
    static real_type const expected_pol_moment[]
        = {-0.034297422127708, 0.79634723763554};

    EXPECT_VEC_SOFT_EQ(expected_dir_moment, dir_moment);
    EXPECT_VEC_SOFT_EQ(expected_pol_moment, pol_moment);
    EXPECT_EQ(12200, rng_engine.count());
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
    auto model = create_model();
    auto builder = this->create_mfp_builder();

    for (auto mat : range(OpticalMaterialId(import_materials().size())))
    {
        model->build_mfps(mat, builder);
    }

    this->check_built_table_exact(
        this->import_models()[import_model_id_.get()].mfp_table,
        builder.grid_ids());
}

//---------------------------------------------------------------------------//
// Check Rayleigh model MFP tables constructed from Einstein-Smoluchowski
// formula based on material data
TEST_F(RayleighModelTest, material_mfp)
{
    static std::vector<std::vector<double>> expected_energies = {
        {1.098177, 1.256172, 1.484130},
        {1.098177, 1.256172, 1.484130},
        {1.098177, 6.812319},
        {1, 2, 5},
        {1.098177, 6.812319},
    };

    static std::vector<std::vector<double>> expected_mfps
        = {{1189584.7068151, 682569.13017288, 343507.60086802},
           {12005096.767467, 6888377.4406869, 3466623.2384762},
           {1189584.7068151, 277.60444893823},
           {11510.805603078, 322.70360179716, 4.230373664558},
           {12005096.767467, 2801.539271218}};

    auto expected_mfp_tables
        = detail::convert_vector_units<detail::ElectronVolt, units::Centimeter>(
            expected_energies, expected_mfps);

    auto model = create_empty_model();
    auto builder = this->create_mfp_builder();

    for (auto mat : range(OpticalMaterialId(import_materials().size())))
    {
        model->build_mfps(mat, builder);
    }

    this->check_built_table_soft(expected_mfp_tables, builder.grid_ids());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
