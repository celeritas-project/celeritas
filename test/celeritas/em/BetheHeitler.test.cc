//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/BetheHeitler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/interactor/BetheHeitlerInteractor.hh"
#include "celeritas/em/model/BetheHeitlerModel.hh"
#include "celeritas/em/process/GammaConversionProcess.hh"
#include "celeritas/mat/ElementView.hh"
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

class BetheHeitlerInteractorTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        auto const& params = *this->particle_params();
        data_.ids.electron = params.find(pdg::electron());
        data_.ids.positron = params.find(pdg::positron());
        data_.ids.gamma = params.find(pdg::gamma());
        data_.electron_mass = params.get(data_.ids.electron).mass();
        data_.enable_lpm = true;

        // Set default particle to photon with energy of 100 MeV
        this->set_inc_particle(pdg::gamma(), MevEnergy{100.0});
        this->set_inc_direction({0, 0, 1});
        this->set_material("Cu-1.0");
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Check change to parent (gamma) track
        EXPECT_EQ(0, interaction.energy.value());
        EXPECT_EQ(Action::absorbed, interaction.action);

        // Check secondaries
        ASSERT_EQ(2, interaction.secondaries.size());
        // Electron
        auto const& electron = interaction.secondaries.front();
        EXPECT_TRUE(electron);
        EXPECT_EQ(data_.ids.electron, electron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  electron.energy.value());
        EXPECT_LT(0, electron.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(electron.direction));
        // Positron
        auto const& positron = interaction.secondaries.back();
        EXPECT_TRUE(positron);
        EXPECT_EQ(data_.ids.positron, positron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  positron.energy.value());
        EXPECT_LT(0, positron.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(positron.direction));

        // Check conservation between primary and secondaries
        // TODO: is momentum known *not* to be conserved?
        // this->check_conservation(interaction);
        this->check_energy_conservation(interaction);
    }

  protected:
    BetheHeitlerData data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(BetheHeitlerInteractorTest, basic)
{
    // Reserve 4 secondaries, two for each sample
    int const num_samples = 4;
    this->resize_secondaries(2 * num_samples);

    // Get views to the current material and element
    auto const material = this->material_track().material_record();
    auto const element = material.element_record(ElementComponentId{0});

    // Create the interactor
    BetheHeitlerInteractor interact(data_,
                                    this->particle_track(),
                                    this->direction(),
                                    this->secondary_allocator(),
                                    material,
                                    element);
    RandomEngine& rng_engine = this->rng();
    // Produce four samples from the original/incident photon
    std::vector<double> angle;
    std::vector<double> energy1;
    std::vector<double> energy2;

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
    double const expected_energy1[] = {
        15.2508794873183, 98.7412722423312, 23.4953328454145, 94.7258588843146};
    double const expected_energy2[] = {
        83.7271226204817, 0.236729865468827, 75.4826692623855, 4.25214322348543};
    double const expected_angle[] = {0.999969298729478,
                                     0.749593336413488,
                                     0.999747408792083,
                                     0.99092640152178};

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

TEST_F(BetheHeitlerInteractorTest, stress_test)
{
    unsigned int const num_samples = 1000;
    std::vector<double> avg_engine_samples;

    // Loop over a set of incident gamma energies
    for (real_type inc_e : {1.5, 5.0, 10.0, 50.0, 100.0, 1e6})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

        RandomEngine& rng_engine = this->rng();
        RandomEngine::size_type num_particles_sampled = 0;

        // Loop over several incident directions
        for (Real3 const& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(2 * num_samples);

            // Get views to the current material and element
            auto const material = this->material_track().material_record();
            auto const element = material.element_record(ElementComponentId{0});

            // Create interactor
            BetheHeitlerInteractor interact(data_,
                                            this->particle_track(),
                                            this->direction(),
                                            this->secondary_allocator(),
                                            material,
                                            element);

            // Loop over many particles
            for (unsigned int i = 0; i < num_samples; ++i)
            {
                Interaction result = interact(rng_engine);
                SCOPED_TRACE(result);
                this->sanity_check(result);
            }
            EXPECT_EQ(2 * num_samples,
                      this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(double(rng_engine.count())
                                     / double(num_particles_sampled));
    }

    // Gold values for average number of calls to RNG
    static double const expected_avg_engine_samples[]
        = {20.127, 24.5935, 24.13, 23.1985, 22.9075, 22.024};
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

TEST_F(BetheHeitlerInteractorTest, distributions)
{
    RandomEngine& rng_engine = this->rng();

    int const num_samples = 10000;
    int const nbins = 10;
    Real3 inc_direction = {0, 0, 1};
    this->set_inc_direction(inc_direction);

    // Get views to the current material and element
    auto const material = this->material_track().material_record();
    auto const element = material.element_record(ElementComponentId{0});

    auto bin_epsilon = [&](real_type inc_energy) -> std::vector<int> {
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_energy});
        this->resize_secondaries(2 * num_samples);

        // Create interactor
        BetheHeitlerInteractor interact(data_,
                                        this->particle_track(),
                                        this->direction(),
                                        this->secondary_allocator(),
                                        material,
                                        element);

        std::vector<int> eps_dist(nbins);

        // Loop over many particles
        for (int i = 0; i < num_samples; ++i)
        {
            Interaction out = interact(rng_engine);
            // Bin the electron reduced energy \epsilon = (E_e + m_e c^2 /
            // E_{\gamma}
            auto const electron = out.secondaries.front();
            double eps = (electron.energy.value() + data_.electron_mass.value())
                         / inc_energy;
            auto eps_bin = static_cast<int>(eps * nbins);
            if (eps_bin >= 0 && eps_bin < nbins)
            {
                ++eps_dist[eps_bin];
            }
        }
        EXPECT_EQ(2 * num_samples, this->secondary_allocator().get().size());
        return eps_dist;
    };

    // 1.5 MeV incident photon
    {
        std::vector<int> eps_dist = bin_epsilon(1.5);
        static int const expected_eps_dist[]
            = {0, 0, 0, 1911, 3054, 3142, 1893, 0, 0, 0};
        EXPECT_VEC_EQ(expected_eps_dist, eps_dist);
    }

    // 100 MeV incident photon
    {
        std::vector<int> eps_dist = bin_epsilon(100);
        static int const expected_eps_dist[]
            = {754, 1109, 1054, 1055, 1010, 1010, 1024, 1055, 1090, 839};
        EXPECT_VEC_EQ(expected_eps_dist, eps_dist);
    }

    // 1 TeV incident photon (LPM effect)
    {
        std::vector<int> eps_dist = bin_epsilon(1e6);
        static int const expected_eps_dist[]
            = {1209, 1073, 911, 912, 844, 881, 903, 992, 1066, 1209};
        EXPECT_VEC_EQ(expected_eps_dist, eps_dist);
    }

    // Interaction threshold energy
    {
        std::vector<int> eps_dist
            = bin_epsilon(2 * data_.electron_mass.value());
        static int const expected_eps_dist[]
            = {0, 0, 0, 0, 0, 10000, 0, 0, 0, 0};
        EXPECT_VEC_EQ(expected_eps_dist, eps_dist);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
