//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBerger.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/detail/SeltzerBergerInteractor.hh"
#include "physics/em/detail/SBPositronXsCorrector.hh"
#include "physics/em/detail/SBEnergyDistribution.hh"
#include "physics/em/SeltzerBergerModel.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/MaterialTrackView.hh"

#include "celeritas_test.hh"
#include "gtest/Main.hh"
#include "base/Algorithms.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "io/SeltzerBergerReader.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::ElementId;
using celeritas::ElementView;
using celeritas::SeltzerBergerModel;
using celeritas::SeltzerBergerReader;
using celeritas::detail::SBEnergyDistribution;
using celeritas::detail::SBPositronXsCorrector;
using celeritas::detail::SeltzerBergerInteractor;
using celeritas::units::AmuMass;
using celeritas::units::MevMass;
namespace constants = celeritas::constants;
namespace pdg       = celeritas::pdg;

using Energy   = celeritas::units::MevEnergy;
using EnergySq = SBEnergyDistribution::EnergySq;

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
        using celeritas::ParticleDef;
        using namespace celeritas::constants;
        using namespace celeritas::units;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

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
        const auto& particles   = *this->particle_params();
        pointers_.ids.electron  = particles.find(pdg::electron());
        pointers_.ids.positron  = particles.find(pdg::positron());
        pointers_.ids.gamma     = particles.find(pdg::gamma());
        pointers_.electron_mass = particles.get(pointers_.ids.electron).mass();

        // Set default particle to incident 1 MeV photon
        this->set_inc_particle(pdg::electron(), MevEnergy{10.0});
        this->set_inc_direction({0, 0, 1});

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.elements  = {{29, AmuMass{63.546}, "Cu"}};
        mat_inp.materials = {
            {1.0 * na_avogadro,
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

        // Construct SeltzerBergerModel and set host pointers
        model_    = std::make_shared<SeltzerBergerModel>(ModelId{0},
                                                      *this->particle_params(),
                                                      *this->material_params(),
                                                      read_element_data);
        pointers_ = model_->host_pointers();

        // Set cutoffs
        CutoffParams::Input           input;
        CutoffParams::MaterialCutoffs material_cutoffs;
        material_cutoffs.push_back({MevEnergy{0.01}, 0.1234});
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
    celeritas::detail::SeltzerBergerNativeRef pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SeltzerBergerTest, sb_tables)
{
    const auto& xs = model_->host_pointers().differential_xs;

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
    const double expected_scaling_frac[] = {
        0.9980342329193, 0.9800123976959, 0.8513338610609, 2.827839668085e-05,
        0.9999449017036, 0.9993355880532, 0.9870854387438, 0.04176283721387,
        0.9999993627271, 0.9999919054025, 0.9997522383667, 0.4174036918268,
        0.9999999935293, 0.9999999173093, 0.9999972926228, 0.8399650995661};
    // clang-format on
    EXPECT_VEC_SOFT_EQ(expected_scaling_frac, scaling_frac);
}

TEST_F(SeltzerBergerTest, sb_energy_dist)
{
    const MevEnergy gamma_cutoff{0.0009};

    const int           num_samples = 8192;
    std::vector<double> max_xs;
    std::vector<double> avg_exit_frac;
    std::vector<double> avg_engine_samples;

    // Note: the first point has a very low cross section compared to
    // ionization so won't be encountered in practice. The differential cross
    // section distribution is much flatter there, so there should be lower
    // rejection. The second point is where the maximum of the differential SB
    // data switches between a high-exit-energy peak and a low-exit-energy
    // peak, which should result in a higher rejection rate. The remaining
    // points are arbitrary.
    for (real_type inc_energy : {0.001, 0.0045, 0.567, 7.89, 89.0, 901.})
    {
        SBEnergyDistribution sample_energy(
            model_->host_pointers(),
            Energy{inc_energy},
            ElementId{0},
            this->density_correction(MaterialId{0}, Energy{inc_energy}),
            gamma_cutoff);
        max_xs.push_back(sample_energy.max_xs());
        double total_exit_energy = 0;

        // Loop over many particles
        RandomEngine& rng_engine = this->rng();
        for (int i = 0; i < num_samples; ++i)
        {
            Energy exit_gamma = sample_energy(rng_engine);
            EXPECT_GT(exit_gamma.value(), gamma_cutoff.value());
            EXPECT_LT(exit_gamma.value(), inc_energy);
            total_exit_energy += exit_gamma.value();
        }

        avg_exit_frac.push_back(total_exit_energy / (num_samples * inc_energy));
        avg_engine_samples.push_back(double(rng_engine.count()) / num_samples);
    }

    // clang-format off
    const double expected_max_xs[] = {2.866525852195, 4.72696244794,
        12.18911946078, 13.93366489719, 13.85758694967, 13.3353235437};
    const double expected_avg_exit_frac[] = {0.9491159324044, 0.4974867596411,
        0.08235370866815, 0.0719988569368, 0.08780979490539, 0.1003040929175};
    const double expected_avg_engine_samples[] = {4.0791015625, 4.06005859375,
        5.13916015625, 4.71923828125, 4.48486328125, 4.40869140625};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_max_xs, max_xs);
    EXPECT_VEC_SOFT_EQ(expected_avg_exit_frac, avg_exit_frac);
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

TEST_F(SeltzerBergerTest, basic)
{
    using celeritas::CutoffView;
    using celeritas::MaterialView;

    // Reserve 2 secondaries, one for each sample
    const int num_samples = 2;
    this->resize_secondaries(num_samples);

    // Sampled element
    const ElementId element{0};

    // Production cuts
    auto                cutoffs = this->cutoff_params()->get(MaterialId{0});
    const MaterialView& material_view = this->material_track().material_view();

    // Create the interactor
    SeltzerBergerInteractor interact(model_->host_pointers(),
                                     this->particle_track(),
                                     this->direction(),
                                     cutoffs,
                                     this->secondary_allocator(),
                                     material_view,
                                     element);
    RandomEngine&           rng_engine = this->rng();

    // Produce two samples from the original/incident photon
    std::vector<double> angles;
    std::vector<double> energies;

    // Loop number of samples
    for (int i : celeritas::range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        angles.push_back(celeritas::dot_product(
            result.direction, result.secondaries.front().direction));
        energies.push_back(result.secondaries[0].energy.value());
    }

    // EXPECT_EQ(2 * num_samples, this->secondary_allocator().get().size());

    // // Note: these are "gold" values based on the host RNG.
    // const double expected_energy1[]
    //     = {16.57248532448, 99.25227118843, 24.00633179151, 95.23685783041};
    // const double expected_energy2[]
    //     = {83.42751467552, 0.7477288115688, 75.99366820849, 4.763142169585};
    // const double expected_angle[]
    //     = {0.9999694782475, 0.9111977209393, 0.9997556894823,
    //     0.9921593039016};

    // EXPECT_VEC_SOFT_EQ(expected_energy1, energy1);
    // EXPECT_VEC_SOFT_EQ(expected_energy2, energy2);
    // EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // // Next sample should fail because we're out of secondary buffer space
    // {
    //     Interaction result = interact(rng_engine);
    //     EXPECT_EQ(0, result.secondaries.size());
    //     EXPECT_EQ(celeritas::Action::failed, result.action);
    // }
}

// TEST_F(SeltzerBergerTest, stress_test)
// {
//     const unsigned int  num_samples = 8;
//     std::vector<double> avg_engine_samples;

//     // Loop over a set of incident gamma energies
//     for (double inc_e : {1.5, 5.0, 10.0, 50.0, 100.0})
//     {
//         SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
//         this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

//         RandomEngine&           rng_engine            = this->rng();
//         RandomEngine::size_type num_particles_sampled = 0;

//         // Loop over several incident directions
//         for (const Real3& inc_dir :
//              {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1,
//              1}})
//         {
//             SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
//             this->set_inc_direction(inc_dir);
//             this->resize_secondaries(2 * num_samples);

//             // Get the ElementView
//             const celeritas::ElementView element(
//                 this->material_track().material_view().element_view(
//                     celeritas::ElementComponentId{0}));

//             // Create interactor
//             BetheHeitlerInteractor interact(pointers_,
//                                             this->particle_track(),
//                                             this->direction(),
//                                             this->secondary_allocator(),
//                                             element);

//             // Loop over many particles
//             for (unsigned int i = 0; i < num_samples; ++i)
//             {
//                 Interaction result = interact(rng_engine);
//                 SCOPED_TRACE(result);
//                 this->sanity_check(result);
//             }
//             EXPECT_EQ(2 * num_samples,
//                       this->secondary_allocator().get().size());
//             num_particles_sampled += num_samples;
//         }
//         avg_engine_samples.push_back(double(rng_engine.count())
//                                      / double(num_particles_sampled));
//     }

//     // Gold values for average number of calls to RNG
//     const double expected_avg_engine_samples[]
//         = {18.375, 23.125, 22.75, 23.3125, 22.5625};
//     EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
// }
