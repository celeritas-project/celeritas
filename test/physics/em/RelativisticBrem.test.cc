//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RelativisticBremTest.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/RelativisticBremModel.hh"
#include "physics/em/detail/RBDiffXsCalculator.hh"
#include "physics/em/detail/RelativisticBremInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "base/Units.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/Units.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::ElementComponentId;
using celeritas::ElementId;
using celeritas::ElementView;
using celeritas::RelativisticBremModel;
using celeritas::detail::RBDiffXsCalculator;
using celeritas::detail::RelativisticBremInteractor;

namespace pdg = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RelativisticBremTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::MatterState;
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        using namespace celeritas::constants;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

        Base::set_particle_params(
            {{"electron",
              pdg::electron(),
              MevMass{0.5109989461},
              ElementaryCharge{-1},
              stable},
             {"positron",
              pdg::positron(),
              MevMass{0.5109989461},
              ElementaryCharge{1},
              stable},
             {"gamma", pdg::gamma(), zero, zero, stable}});
        const auto& particles = *this->particle_params();

        // Set up shared material data
        MaterialParams::Input mi;
        mi.elements  = {{82, AmuMass{207.2}, "Pb"}};
        mi.materials = {{0.05477 * na_avogadro,
                         293.15,
                         MatterState::solid,
                         {{ElementId{0}, 1.0}},
                         "Pb"}};

        // Set default material to potassium
        this->set_material_params(mi);
        this->set_material("Pb");

        // Set default particle to incident 25 GeV electron
        this->set_inc_particle(pdg::positron(), MevEnergy{25000});
        this->set_inc_direction({0, 0, 1});

        // Construct RelativisticBremModel and save the host data reference
        model_ = std::make_shared<RelativisticBremModel>(
            ModelId{0}, particles, *this->material_params(), false);
        data_ = model_->host_ref();

        // Construct RelativisticBremModel and save the host data reference
        model_lpm_ = std::make_shared<RelativisticBremModel>(
            ModelId{0}, particles, *this->material_params(), true);
        data_lpm_ = model_lpm_->host_ref();

        // Set cutoffs: photon energy thresholds and range cut for Pb
        CutoffParams::Input           input;
        CutoffParams::MaterialCutoffs material_cutoffs;
        material_cutoffs.push_back({MevEnergy{0.0945861}, 0.07});
        input.materials = this->material_params();
        input.particles = this->particle_params();
        input.cutoffs.insert({pdg::gamma(), material_cutoffs});
        this->set_cutoff_params(input);
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

        // Check secondaries (bremsstrahlung photon)
        ASSERT_EQ(1, interaction.secondaries.size());

        const auto& gamma = interaction.secondaries.front();
        EXPECT_TRUE(gamma);
        EXPECT_EQ(data_.ids.gamma, gamma.particle_id);

        // Check conservation
        this->check_conservation(interaction);
    }

  protected:
    std::shared_ptr<RelativisticBremModel>       model_;
    celeritas::detail::RelativisticBremNativeRef data_;
    std::shared_ptr<RelativisticBremModel>       model_lpm_;
    celeritas::detail::RelativisticBremNativeRef data_lpm_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(RelativisticBremTest, dxsec)
{
    const real_type all_energy[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};

    // Production cuts
    auto material_view = this->material_track().material_view();

    // Create the differential cross section
    RBDiffXsCalculator dxsec_lpm(model_lpm_->host_ref(),
                                 this->particle_track(),
                                 material_view,
                                 ElementComponentId{0});

    // Create the differential cross section
    RBDiffXsCalculator dxsec(model_->host_ref(),
                             this->particle_track(),
                             material_view,
                             ElementComponentId{0});

    // Calculate cross section values at ten photon energy (MevEnergy) points
    std::vector<double> dxsec_value_lpm;
    std::vector<double> dxsec_value;

    for (real_type energy : all_energy)
    {
        real_type result = dxsec_lpm(MevEnergy{energy});
        dxsec_value_lpm.push_back(result);

        result = dxsec(MevEnergy{energy});
        dxsec_value.push_back(result);
    }

    // Note: these are "gold" differential cross sections by the photon energy.
    const double expected_dxsec_lpm[] = {3.1589268205686,
                                         2.24021800150128,
                                         1.76771499385005,
                                         1.93837223714831,
                                         2.31473664706576,
                                         2.8953506143604,
                                         3.2720266599198,
                                         3.50840669766741,
                                         3.48017265095914,
                                         3.41228707307214};

    const double expected_dxsec[] = {3.55000253342095,
                                     3.54986051043622,
                                     3.54943449138746,
                                     3.54872462599092,
                                     3.54730551901548,
                                     3.54305318862896,
                                     3.53598260644102,
                                     3.52190382370326,
                                     3.48016652710843,
                                     3.41226786120072};

    EXPECT_VEC_SOFT_EQ(expected_dxsec_lpm, dxsec_value_lpm);
    EXPECT_VEC_SOFT_EQ(expected_dxsec, dxsec_value);
}

TEST_F(RelativisticBremTest, basic_without_lpm)
{
    const int num_samples = 4;

    // Reserve  num_samples secondaries;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().material_view();
    auto cutoffs       = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    RelativisticBremInteractor interact(model_->host_ref(),
                                        this->particle_track(),
                                        this->direction(),
                                        cutoffs,
                                        this->secondary_allocator(),
                                        material_view,
                                        ElementComponentId{0});

    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original incident angle/energy
    std::vector<double> angle;
    std::vector<double> energy;

    for (int i : celeritas::range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        energy.push_back(result.secondaries[0].energy.value());
        angle.push_back(celeritas::dot_product(
            result.direction, result.secondaries.back().direction));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
    EXPECT_DOUBLE_EQ(double(rng_engine.count()) / num_samples, 12);

    // Note: these are "gold" values based on the host RNG.

    const double expected_energy[] = {
        9.7121539090503, 16.1109589071687, 7.48863745059463, 8338.70226190511};

    const double expected_angle[] = {0.999999999858782,
                                     0.999999999999921,
                                     0.999999976998416,
                                     0.999999998137601};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}

TEST_F(RelativisticBremTest, basic_with_lpm)
{
    const int num_samples = 4;

    // Reserve  num_samples secondaries;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().material_view();
    auto cutoffs       = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    RelativisticBremInteractor interact(model_lpm_->host_ref(),
                                        this->particle_track(),
                                        this->direction(),
                                        cutoffs,
                                        this->secondary_allocator(),
                                        material_view,
                                        ElementComponentId{0});

    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original incident angle/energy
    std::vector<double> angle;
    std::vector<double> energy;

    for (int i : celeritas::range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        energy.push_back(result.secondaries[0].energy.value());
        angle.push_back(celeritas::dot_product(
            result.direction, result.secondaries.back().direction));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.

    const double expected_energy[] = {
        18872.4157243063, 43.6117832245235, 4030.31152398788, 217.621447606391};

    const double expected_angle[] = {0.999999971800136,
                                     0.999999999587026,
                                     0.999999999683752,
                                     0.999999999474844};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);
}

TEST_F(RelativisticBremTest, stress_with_lpm)
{
    const int num_samples = 1000;

    // Reserve  num_samples secondaries;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().material_view();
    auto cutoffs       = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    RelativisticBremInteractor interact(model_lpm_->host_ref(),
                                        this->particle_track(),
                                        this->direction(),
                                        cutoffs,
                                        this->secondary_allocator(),
                                        material_view,
                                        ElementComponentId{0});

    RandomEngine& rng_engine = this->rng();

    // Produce samples from the original incident angle/energy
    real_type average_energy{0};
    Real3     average_angle{0, 0, 0};

    for (int i : celeritas::range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);

        average_energy += result.secondaries[0].energy.value();
        average_angle[0] += result.secondaries[0].direction[0];
        average_angle[1] += result.secondaries[0].direction[1];
        average_angle[2] += result.secondaries[0].direction[2];

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);
    }
    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    EXPECT_DOUBLE_EQ(average_energy / num_samples, 2932.1072998587733);
    EXPECT_DOUBLE_EQ(average_angle[0] / num_samples, 3.3286986548662216e-06);
    EXPECT_DOUBLE_EQ(average_angle[1] / num_samples, 1.3067055198983571e-06);
    EXPECT_DOUBLE_EQ(average_angle[2] / num_samples, 0.99999999899845182);
}
