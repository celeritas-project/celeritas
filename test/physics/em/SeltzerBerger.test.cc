//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBerger.test.cc
//---------------------------------------------------------------------------//
#include "base/Algorithms.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "io/SeltzerBergerReader.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/Units.hh"
#include "physics/em/BremsstrahlungProcess.hh"
#include "physics/em/SeltzerBergerModel.hh"
#include "physics/em/detail/SBEnergyDistribution.hh"
#include "physics/em/detail/SBPositronXsCorrector.hh"
#include "physics/em/detail/SeltzerBergerInteractor.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/material/MaterialView.hh"

#include "../InteractionIO.hh"
#include "../InteractorHostTestBase.hh"
#include "celeritas_test.hh"
#include "gtest/Main.hh"

using celeritas::BremsstrahlungProcess;
using celeritas::ElementComponentId;
using celeritas::ElementId;
using celeritas::ElementView;
using celeritas::SeltzerBergerModel;
using celeritas::SeltzerBergerReader;
using celeritas::detail::SBElectronXsCorrector;
using celeritas::detail::SBEnergyDistHelper;
using celeritas::detail::SBEnergyDistribution;
using celeritas::detail::SBPositronXsCorrector;
using celeritas::detail::SeltzerBergerInteractor;
using celeritas::units::AmuMass;
using celeritas::units::MevMass;
namespace constants = celeritas::constants;
namespace pdg       = celeritas::pdg;

using Energy   = celeritas::units::MevEnergy;
using EnergySq = SBEnergyDistHelper::EnergySq;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SeltzerBergerTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::MatterState;
        using celeritas::ParticleRecord;
        using namespace celeritas::constants;
        using namespace celeritas::units;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleRecord::stable_decay_constant();

        // Set up shared particle data
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
        data_.ids.electron    = particles.find(pdg::electron());
        data_.ids.positron    = particles.find(pdg::positron());
        data_.ids.gamma       = particles.find(pdg::gamma());
        data_.electron_mass   = particles.get(data_.ids.electron).mass();

        // Set default particle to incident 1 MeV photon
        this->set_inc_particle(pdg::electron(), MevEnergy{1.0});
        this->set_inc_direction({0, 0, 1});

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.elements  = {{29, AmuMass{63.546}, "Cu"}};
        mat_inp.materials = {
            {0.141 * na_avogadro,
             293.0,
             MatterState::solid,
             {{ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(mat_inp);
        this->set_material("Cu");

        // Set up Seltzer-Berger cross section data
        std::string         data_path = this->test_data_path("physics/em", "");
        SeltzerBergerReader read_element_data(data_path.c_str());

        // Construct SeltzerBergerModel and set host data
        model_ = std::make_shared<SeltzerBergerModel>(ModelId{0},
                                                      *this->particle_params(),
                                                      *this->material_params(),
                                                      read_element_data);
        data_  = model_->host_ref();

        // Set cutoffs
        CutoffParams::Input           input;
        CutoffParams::MaterialCutoffs material_cutoffs;
        material_cutoffs.push_back({MevEnergy{0.02064384}, 0.07});
        input.materials = this->material_params();
        input.particles = this->particle_params();
        input.cutoffs.insert({pdg::gamma(), material_cutoffs});
        this->set_cutoff_params(input);
    }

    EnergySq density_correction(MaterialId matid, Energy e) const
    {
        CELER_EXPECT(matid);
        CELER_EXPECT(e > celeritas::zero_quantity());
        using celeritas::ipow;
        using namespace celeritas::constants;

        auto           mat    = this->material_params()->get(matid);
        constexpr auto migdal = 4 * pi * r_electron
                                * ipow<2>(lambdabar_electron);

        real_type density_factor = mat.electron_density() * migdal;
        return EnergySq{density_factor * ipow<2>(e.value())};
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);
    }

  protected:
    std::shared_ptr<SeltzerBergerModel>       model_;
    celeritas::detail::SeltzerBergerRef       data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SeltzerBergerTest, sb_tables)
{
    const auto& xs = model_->host_ref().differential_xs;

    // The tables should just have the one element (copper). The values of the
    // arguments have been calculated from the g4emlow@7.13 dataset.
    ASSERT_EQ(1, xs.elements.size());

    auto               argmax = xs.sizes[xs.elements[ElementId{0}].argmax];
    const unsigned int expected_argmax[]
        = {31, 31, 31, 30, 30, 7, 7, 6, 5, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0,
           0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_EQ(argmax, expected_argmax);
}

TEST_F(SeltzerBergerTest, sb_positron_xs_scaling)
{
    const ParticleParams& pp        = *this->particle_params();
    const MevMass     positron_mass = pp.get(pp.find(pdg::positron())).mass();
    const MevEnergy   gamma_cutoff{0.01};
    const ElementView el = this->material_params()->get(ElementId{0});

    std::vector<real_type> scaling_frac;

    for (real_type inc_energy : {1.0, 10.0, 100., 1000.})
    {
        SBPositronXsCorrector scale_xs(
            positron_mass, el, gamma_cutoff, MevEnergy{inc_energy});
        for (real_type sampled_efrac : {.10001, .5, .9, .9999})
        {
            real_type exit_energy = sampled_efrac * inc_energy;
            scaling_frac.push_back(scale_xs(MevEnergy{exit_energy}));
        }
    }
    // clang-format off
    static const double expected_scaling_frac[] = {
        0.98771267862086, 0.88085886234621, 0.36375147691123, 2.6341925633236e-29,
        0.99965385757708, 0.99583269657665, 0.92157316225919, 2.1585790781929e-09,
        0.99999599590292, 0.99994914123134, 0.99844428624414, 0.0041293798201,
        0.99999995934326, 0.99999948043882, 0.99998298916928, 0.33428689072689};
    // clang-format on
    EXPECT_VEC_SOFT_EQ(expected_scaling_frac, scaling_frac);
}

TEST_F(SeltzerBergerTest, sb_energy_dist)
{
    const MevEnergy gamma_cutoff{0.0009};

    const int           num_samples = 8192;
    std::vector<double> max_xs;
    std::vector<double> xs_zero;
    std::vector<double> avg_exit_frac;
    std::vector<double> avg_engine_samples;

    auto sample_many = [&](real_type inc_energy, auto& sample_energy) {
        double        total_exit_energy = 0;
        RandomEngine& rng_engine        = this->rng();
        for (int i = 0; i < num_samples; ++i)
        {
            Energy exit_gamma = sample_energy(rng_engine);
            EXPECT_GT(exit_gamma.value(), gamma_cutoff.value());
            EXPECT_LT(exit_gamma.value(), inc_energy);
            total_exit_energy += exit_gamma.value();
        }

        avg_exit_frac.push_back(total_exit_energy / (num_samples * inc_energy));
        avg_engine_samples.push_back(double(rng_engine.count()) / num_samples);
    };

    // Note: the first point has a very low cross section compared to
    // ionization so won't be encountered in practice. The differential cross
    // section distribution is much flatter there, so there should be lower
    // rejection. The second point is where the maximum of the differential SB
    // data switches between a high-exit-energy peak and a low-exit-energy
    // peak, which should result in a higher rejection rate. The remaining
    // points are arbitrary.
    for (real_type inc_energy : {0.001, 0.0045, 0.567, 7.89, 89.0, 901.})
    {
        SBEnergyDistHelper edist_helper(
            model_->host_ref().differential_xs,
            Energy{inc_energy},
            ElementId{0},
            this->density_correction(MaterialId{0}, Energy{inc_energy}),
            gamma_cutoff);
        max_xs.push_back(edist_helper.max_xs().value());
        xs_zero.push_back(edist_helper.xs_zero().value());

        SBEnergyDistribution<SBElectronXsCorrector> sample_energy(edist_helper,
                                                                  {});
        // Loop over many particles
        sample_many(inc_energy, sample_energy);
    }

    {
        real_type inc_energy = 7.89;

        SBEnergyDistHelper edist_helper(
            model_->host_ref().differential_xs,
            Energy{inc_energy},
            ElementId{0},
            this->density_correction(MaterialId{0}, Energy{inc_energy}),
            gamma_cutoff);

        // Sample with a "correction" that's constant, which shouldn't change
        // sampling efficiency or expected value correction
        {
            struct ScaleXs
            {
                using Xs = celeritas::Quantity<celeritas::units::Millibarn>;

                real_type operator()(Energy) const { return 0.5; }

                Xs max_xs(const SBEnergyDistHelper& helper) const
                {
                    return helper.calc_xs(MevEnergy{0.0009});
                }
            };

            SBEnergyDistribution<ScaleXs> sample_energy(edist_helper,
                                                        ScaleXs{});

            // Loop over many particles
            sample_many(inc_energy, sample_energy);
        }

        // Sample with the positron XS correction
        {
            const ParticleParams& pp = *this->particle_params();

            SBEnergyDistribution<SBPositronXsCorrector> sample_energy(
                edist_helper,
                {pp.get(pp.find(pdg::positron())).mass(),
                 this->material_params()->get(ElementId{0}),
                 gamma_cutoff,
                 Energy{inc_energy}});

            // Loop over many particles
            sample_many(inc_energy, sample_energy);
        }
    }

    // clang-format off
    const double expected_max_xs[] = {2.866525852195, 4.72696244794,
        12.18911946078, 13.93366489719, 13.85758694967, 13.3353235437};
    const double expected_xs_zero[] = {1.98829818915769, 4.40320232447369,
        12.18911946078, 13.93366489719, 13.85758694967, 13.3353235437};
    const double expected_avg_exit_frac[] = {0.949115932248866,
        0.497486662164049, 0.082127972143285, 0.0645177016233406, 
        0.0774717918229646, 0.0891340819129683, 0.0639090949553034, 
        0.0642877319142647};
    const double expected_avg_engine_samples[] = {4.0791015625, 4.06005859375,
	5.134765625, 4.65625, 4.43017578125, 4.35693359375, 9.3681640625,
        4.65478515625};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_max_xs, max_xs);
    EXPECT_VEC_SOFT_EQ(expected_xs_zero, xs_zero);
    EXPECT_VEC_SOFT_EQ(expected_avg_exit_frac, avg_exit_frac);
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

TEST_F(SeltzerBergerTest, basic)
{
    using celeritas::MaterialView;

    // Reserve 4 secondaries, one for each sample
    const int num_samples = 4;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().make_material_view();
    auto cutoffs       = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    SeltzerBergerInteractor interact(model_->host_ref(),
                                     this->particle_track(),
                                     this->direction(),
                                     cutoffs,
                                     this->secondary_allocator(),
                                     material_view,
                                     ElementComponentId{0});
    RandomEngine&           rng_engine = this->rng();

    // Produce two samples from the original/incident photon
    std::vector<double> angle;
    std::vector<double> energy;

    // Loop number of samples
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
            result.direction, result.secondaries.front().direction));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    const double expected_angle[]  = {0.959441513277674,
                                     0.994350429950924,
                                     0.968866136008621,
                                     0.961582855967571};
    const double expected_energy[] = {0.0349225070114679,
                                      0.0316182310804369,
                                      0.0838794010486177,
                                      0.106195186929141};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}

TEST_F(SeltzerBergerTest, stress_test)
{
    using celeritas::MaterialView;

    const int           num_samples = 1e4;
    std::vector<double> avg_engine_samples;

    // Views
    auto cutoffs       = this->cutoff_params()->get(MaterialId{0});
    auto material_view = this->material_track().make_material_view();

    // Loop over a set of incident gamma energies
    for (auto particle : {pdg::electron(), pdg::positron()})
    {
        for (double inc_e : {1.5, 5.0, 10.0, 50.0, 100.0})
        {
            SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
            this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

            RandomEngine&           rng_engine            = this->rng();
            RandomEngine::size_type num_particles_sampled = 0;

            // Loop over several incident directions
            for (const Real3& inc_dir : {Real3{0, 0, 1},
                                         Real3{1, 0, 0},
                                         Real3{1e-9, 0, 1},
                                         Real3{1, 1, 1}})
            {
                this->set_inc_direction(inc_dir);
                this->resize_secondaries(num_samples);

                // Create interactor
                this->set_inc_particle(particle, MevEnergy{inc_e});
                SeltzerBergerInteractor interact(model_->host_ref(),
                                                 this->particle_track(),
                                                 this->direction(),
                                                 cutoffs,
                                                 this->secondary_allocator(),
                                                 material_view,
                                                 ElementComponentId{0});

                // Loop over many particles
                for (unsigned int i = 0; i < num_samples; ++i)
                {
                    Interaction result = interact(rng_engine);
                    this->sanity_check(result);
                }
                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(double(rng_engine.count())
                                         / double(num_particles_sampled));
        }
    }

    // Gold values for average number of calls to RNG
    static const double expected_avg_engine_samples[] = {14.088,
                                                         13.2402,
                                                         12.9641,
                                                         12.5832,
                                                         12.4988,
                                                         14.2108,
                                                         13.254,
                                                         12.9431,
                                                         12.5952,
                                                         12.4888};

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}
