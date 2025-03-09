//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/EPlusGG.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/EPlusGGInteractor.hh"
#include "celeritas/em/xs/EPlusGGMacroXsCalculator.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/InteractionIO.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class EPlusGGInteractorTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        auto const& params = *this->particle_params();
        data_.positron = params.find(pdg::positron());
        data_.gamma = params.find(pdg::gamma());
        data_.electron_mass = params.get(params.find(pdg::electron())).mass();

        // Set default particle to incident 10 MeV positron
        this->set_inc_particle(pdg::positron(), MevEnergy{10});
        this->set_inc_direction({0, 0, 1});
        this->set_material("K");
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Check change to parent track
        EXPECT_EQ(Action::absorbed, interaction.action);
        EXPECT_EQ(0, interaction.energy.value());

        // Check secondaries (two photons)
        ASSERT_EQ(2, interaction.secondaries.size());

        auto const& gamma1 = interaction.secondaries.front();
        EXPECT_TRUE(gamma1);
        EXPECT_EQ(data_.gamma, gamma1.particle_id);

        EXPECT_GT(this->particle_track().energy().value()
                      + 2 * data_.electron_mass.value(),
                  gamma1.energy.value());
        EXPECT_LT(0, gamma1.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(gamma1.direction));

        auto const& gamma2 = interaction.secondaries.back();
        EXPECT_TRUE(gamma2);
        EXPECT_EQ(data_.gamma, gamma2.particle_id);
        EXPECT_GT(this->particle_track().energy().value()
                      + 2 * data_.electron_mass.value(),
                  gamma2.energy.value());
        EXPECT_LT(0, gamma2.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(gamma2.direction));
    }

  protected:
    EPlusGGData data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(EPlusGGInteractorTest, basic)
{
    int const num_samples = 4;

    // Reserve  num_samples*2 secondaries;
    this->resize_secondaries(num_samples * 2);

    // Create the interactor
    EPlusGGInteractor interact(data_,
                               this->particle_track(),
                               this->direction(),
                               this->secondary_allocator());
    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original incident angle/energy
    std::vector<real_type> angle;
    std::vector<real_type> energy1;
    std::vector<real_type> energy2;

    for (int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        angle.push_back(dot_product(result.secondaries.front().direction,
                                    result.secondaries.back().direction));
        energy1.push_back(result.secondaries[0].energy.value());
        energy2.push_back(result.secondaries[1].energy.value());
    }

    EXPECT_EQ(2 * num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    real_type const expected_energy1[] = {0.432998201097281,
                                          0.595079388679535,
                                          0.527077606856619,
                                          0.266407915356902};

    real_type const expected_energy2[] = {
        10.5889996911027, 10.4269185035205, 10.4949202853434, 10.7555899768431};

    real_type const expected_angle[] = {-0.18912233467373,
                                        0.148337256571676,
                                        0.0320262042116524,
                                        -0.963881322481829};

    EXPECT_VEC_SOFT_EQ(expected_energy1, energy1);
    EXPECT_VEC_SOFT_EQ(expected_energy2, energy2);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(EPlusGGInteractorTest, at_rest)
{
    this->set_inc_direction({1, 0, 0});
    this->set_inc_particle(pdg::positron(), zero_quantity());
    int const num_samples = 4;

    // Reserve  num_samples*2 secondaries;
    this->resize_secondaries(num_samples * 2);

    // Create the interactor
    EPlusGGInteractor interact(data_,
                               this->particle_track(),
                               this->direction(),
                               this->secondary_allocator());
    RandomEngine& rng_engine = this->rng();

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        ASSERT_EQ(2, result.secondaries.size());
        EXPECT_SOFT_EQ(-1,
                       dot_product(result.secondaries[0].direction,
                                   result.secondaries[1].direction));

        EXPECT_SOFT_EQ(data_.electron_mass.value(),
                       result.secondaries[0].energy.value());
        EXPECT_SOFT_EQ(data_.electron_mass.value(),
                       result.secondaries[1].energy.value());
    }
}

TEST_F(EPlusGGInteractorTest, stress_test)
{
    int const num_samples = 8192;
    std::vector<real_type> avg_engine_samples;

    for (real_type inc_e : {0.0, 0.01, 1.0, 10.0, 1000.0})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::positron(), MevEnergy{inc_e});

        RandomEngine& rng_engine = this->rng();
        RandomEngine::size_type num_particles_sampled = 0;

        // Loop over several incident directions
        for (Real3 const& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(2 * num_samples);

            // Create interactor
            EPlusGGInteractor interact(data_,
                                       this->particle_track(),
                                       this->direction(),
                                       this->secondary_allocator());

            // Loop over many particles
            for (int i = 0; i < num_samples; ++i)
            {
                Interaction result = interact(rng_engine);
                // SCOPED_TRACE(result);
                this->sanity_check(result);
            }
            EXPECT_EQ(2 * num_samples,
                      this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(real_type(rng_engine.count())
                                     / real_type(num_particles_sampled));
    }

    // PRINT_EXPECTED(avg_engine_samples);
    // Gold values for average number of calls to RNG
    real_type const expected_avg_engine_samples[]
        = {4, 9.8341064453125, 7.19775390625, 6.960205078125, 6.5401611328125};
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

TEST_F(EPlusGGInteractorTest, macro_xs)
{
    using units::MevEnergy;

    auto material = this->material_track().material_record();
    EPlusGGMacroXsCalculator calc_macro_xs(data_, material);

    int num_vals = 20;
    real_type loge_min = std::log(1.e-4);
    real_type loge_max = std::log(1.e6);
    real_type delta = (loge_max - loge_min) / (num_vals - 1);
    real_type loge = loge_min;

    std::vector<real_type> energy;
    std::vector<real_type> macro_xs;

    // Loop over energies
    for (int i = 0; i < num_vals; ++i)
    {
        real_type e = std::exp(loge);
        energy.push_back(e);
        macro_xs.push_back(
            native_value_to<units::InvCmXs>(calc_macro_xs(MevEnergy{e})).value());
        loge += delta;
    }
    real_type const expected_macro_xs[]
        = {0.001443034416941,  0.0007875334997718, 0.0004301446502063,
           0.0002355766377589, 0.0001301463511539, 7.376415204169e-05,
           4.419813786948e-05, 2.746581269388e-05, 1.508499252627e-05,
           6.80154666357e-06,  2.782643662379e-06, 1.083362674122e-06,
           4.039064800964e-07, 1.451975852737e-07, 5.07363090171e-08,
           1.734848791099e-08, 5.833443676789e-09, 1.93572917075e-09,
           6.355265134801e-10, 2.068312058021e-10};
    EXPECT_VEC_SOFT_EQ(expected_macro_xs, macro_xs);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
