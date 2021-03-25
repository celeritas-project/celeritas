//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabha.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/detail/MollerBhabhaInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/base/CutoffView.hh"

using celeritas::Action;
using celeritas::CutoffView;
using celeritas::dot_product;
using celeritas::normalize_direction;
using celeritas::ParticleCutoff;
using celeritas::detail::MollerBhabhaInteractor;
using celeritas::units::AmuMass;
using celeritas::units::MevEnergy;
namespace constants = celeritas::constants;
namespace pdg       = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MollerBhabhaInteractorTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // Particles needed by interactor
        Base::set_particle_params({{"electron",
                                    pdg::electron(),
                                    MevMass{0.5109989461},
                                    ElementaryCharge{-1},
                                    stable},

                                   {"positron",
                                    pdg::positron(),
                                    MevMass{0.5109989461},
                                    ElementaryCharge{1},
                                    stable}});

        // Setup MaterialView
        MaterialParams::Input inp;
        inp.elements  = {{29, AmuMass{63.546}, "Cu"}};
        inp.materials = {
            {1.0 * constants::na_avogadro,
             293.0,
             celeritas::MatterState::solid,
             {{celeritas::ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(inp);
        this->set_material("Cu");

        // Set basic CutoffParams (no cuts)
        CutoffParams::Input cutoff_inp;
        cutoff_inp.materials = this->material_params();
        cutoff_inp.particles = this->particle_params();
        this->set_cutoff_params(cutoff_inp);

        // Set MollerBhabhaPointers
        const auto& params    = this->particle_params();
        pointers_.electron_id = params->find(pdg::electron());
        pointers_.positron_id = params->find(pdg::positron());
        pointers_.electron_mass_c_sq
            = params->get(pointers_.electron_id).mass().value();
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(interaction.direction));
        EXPECT_EQ(celeritas::Action::scattered, interaction.action);

        // Check secondaries
        ASSERT_EQ(1, interaction.secondaries.size());
        const auto& electron = interaction.secondaries.front();
        EXPECT_TRUE(electron);
        EXPECT_EQ(pointers_.electron_id, electron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  electron.energy.value());
        EXPECT_LT(0, electron.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(electron.direction));

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
    }

  protected:
    celeritas::detail::MollerBhabhaPointers pointers_;
};

struct SampleInit
{
    celeritas::real_type energy; //!< MeV
    celeritas::Real3     dir;
};

struct SampleResult
{
    std::vector<double> inc_exit_cost;
    std::vector<double> inc_exit_e;
    std::vector<double> inc_edep;
    std::vector<double> sec_cost;
    std::vector<double> sec_e;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(MollerBhabhaInteractorTest, basic)
{
    // Sample 4 Moller and 4 Bhabha interactors
    this->resize_secondaries(8);
    RandomEngine& rng_engine = this->rng();

    // Sampled results
    SampleResult m_results, b_results;

    // clang-format off
    // Incident energy [MeV] and unnormalized direction
    const SampleInit samples[] = {{1,   {5, 5, 5}},
                                  {10,  {-3, 7, 10}},
                                  {1e3, {1, -10, 5}},
                                  {1e5, {3, 7, -6}}};
    // clang-format on

    CutoffView cutoff_view(
        this->cutoff_params()->host_pointers(), ParticleId{0}, MaterialId{0});

    for (const SampleInit& init : samples)
    {
        Real3 dir = init.dir;
        normalize_direction(&dir);
        this->set_inc_direction(dir);

        for (auto p : {pdg::electron(), pdg::positron()})
        {
            this->set_inc_particle(p, MevEnergy{init.energy});

            MollerBhabhaInteractor mb_interact(pointers_,
                                               this->particle_track(),
                                               cutoff_view,
                                               dir,
                                               this->secondary_allocator());

            Interaction result = mb_interact(rng_engine);
            this->sanity_check(result);
            const Secondary& sec = result.secondaries.front();

            SampleResult& r = (p == pdg::electron() ? m_results : b_results);
            r.inc_exit_cost.push_back(dot_product(result.direction, dir));
            r.inc_exit_e.push_back(result.energy.value());
            r.inc_edep.push_back(result.energy_deposition.value());
            r.sec_cost.push_back(dot_product(sec.direction, dir));
            r.sec_e.push_back(sec.energy.value());
        }
    }

    //// Moller
    // Gold values based on the host rng. Energies are in MeV
    const double expected_m_inc_exit_cost[]
        = {0.9963292221847, 0.9999872239933, 0.9999999909221, 0.9999999999995};
    const double expected_m_inc_exit_e[]
        = {0.9856047891889, 9.997244984116, 999.9822171745, 99999.99056267};
    const double expected_m_inc_edep[] = {0, 0, 0, 0};
    const double expected_m_sec_cost[] = {
        0.1675857575233, 0.05443555255527, 0.1308432475718, 0.09565458549568};
    const double expected_m_sec_e[] = {
        0.01439521081106, 0.002755015883891, 0.0177828255398, 0.009437329257737};

    //// Bhabha
    // Gold values based on the host rng. Energies are in MeV
    const double expected_b_inc_exit_cost[]
        = {0.9997453107903, 0.999994190499, 0.9999999989865, 0.9999999999999};
    const double expected_b_inc_exit_e[]
        = {0.9989928374838, 9.998747065072, 999.9980145461, 99999.99864983};
    const double expected_b_inc_edep[] = {0, 0, 0, 0};
    const double expected_b_sec_cost[] = {
        0.04461708949452, 0.03673697288345, 0.04405602044159, 0.03632326062515};
    const double expected_b_sec_e[] = {0.001007162516187,
                                       0.001252934927768,
                                       0.001985453873814,
                                       0.001350170413359};

    //// Moller
    EXPECT_VEC_SOFT_EQ(expected_m_inc_exit_cost, m_results.inc_exit_cost);
    EXPECT_VEC_SOFT_EQ(expected_m_inc_exit_e, m_results.inc_exit_e);
    EXPECT_VEC_SOFT_EQ(expected_m_inc_edep, m_results.inc_edep);
    EXPECT_VEC_SOFT_EQ(expected_m_sec_cost, m_results.sec_cost);
    EXPECT_VEC_SOFT_EQ(expected_m_sec_e, m_results.sec_e);
    //// Bhabha
    EXPECT_VEC_SOFT_EQ(expected_b_inc_exit_cost, b_results.inc_exit_cost);
    EXPECT_VEC_SOFT_EQ(expected_b_inc_exit_e, b_results.inc_exit_e);
    EXPECT_VEC_SOFT_EQ(expected_b_inc_edep, b_results.inc_edep);
    EXPECT_VEC_SOFT_EQ(expected_b_sec_cost, b_results.sec_cost);
    EXPECT_VEC_SOFT_EQ(expected_b_sec_e, b_results.sec_e);
}

//---------------------------------------------------------------------------//
TEST_F(MollerBhabhaInteractorTest, cutoff_1MeV)
{
    // Sample 4 Moller and 4 Bhabha interactors
    this->resize_secondaries(8);
    RandomEngine& rng_engine = this->rng();

    // Sampled results
    SampleResult m_results, b_results;

    // clang-format off
    // Incident energy [MeV] and unnormalized direction
    const SampleInit samples[] = {{10,   {5, 5, 5}},
                                  {1e2,  {-3, 7, 10}},
                                  {1e3, {1, -10, 5}},
                                  {1e5, {3, 7, -6}}};
    // clang-format on

    // Create CutoffParams with a 1 MeV electron cutoff (range not needed)
    CutoffParams::MaterialCutoffs material_cutoffs;
    material_cutoffs.push_back({MevEnergy{1}, 0});

    CutoffParams::Input cutoff_inp;
    cutoff_inp.materials = this->material_params();
    cutoff_inp.particles = this->particle_params();
    cutoff_inp.cutoffs.insert({pdg::electron(), material_cutoffs});
    this->set_cutoff_params(cutoff_inp);

    CutoffView cutoff_view(
        this->cutoff_params()->host_pointers(), ParticleId{0}, MaterialId{0});

    for (const SampleInit& init : samples)
    {
        Real3 dir = init.dir;
        normalize_direction(&dir);
        this->set_inc_direction(dir);

        for (auto p : {pdg::electron(), pdg::positron()})
        {
            this->set_inc_particle(p, MevEnergy{init.energy});

            MollerBhabhaInteractor mb_interact(pointers_,
                                               this->particle_track(),
                                               cutoff_view,
                                               dir,
                                               this->secondary_allocator());

            Interaction result = mb_interact(rng_engine);
            this->sanity_check(result);
            const Secondary& sec = result.secondaries.front();

            SampleResult& r = (p == pdg::electron() ? m_results : b_results);
            r.inc_exit_cost.push_back(dot_product(result.direction, dir));
            r.inc_exit_e.push_back(result.energy.value());
            r.inc_edep.push_back(result.energy_deposition.value());
            r.sec_cost.push_back(dot_product(sec.direction, dir));
            r.sec_e.push_back(sec.energy.value());
        }
    }

    //// Moller
    // Gold values based on the host rng. Energies are in MeV
    const double expected_m_inc_exit_cost[]
        = {0.9784675127353, 0.9997401875592, 0.9999953862586, 0.9999999997589};
    const double expected_m_inc_exit_e[]
        = {6.75726441249, 95.11275692125, 991.0427997072, 99995.28168559};
    const double expected_m_inc_edep[] = {0, 0, 0, 0};
    const double expected_m_sec_cost[]
        = {0.9154612855963, 0.91405872098, 0.9478947756541, 0.9066254320384};
    const double expected_m_sec_e[]
        = {3.24273558751, 4.887243078746, 8.957200292789, 4.718314414109};

    //// Bhabha
    // Gold values based on the host rng. Energies are in MeV
    const double expected_b_inc_exit_cost[]
        = {0.9774788335858, 0.9999472046111, 0.9999992012865, 0.999999999931};
    const double expected_b_inc_exit_e[]
        = {6.654742369665, 98.96696134497, 998.4378016843, 99998.64983431};
    const double expected_b_inc_edep[] = {0, 0, 0, 0};
    const double expected_b_sec_cost[]
        = {0.9188415916986, 0.7126175077086, 0.777906053136, 0.7544377929863};
    const double expected_b_sec_e[]
        = {3.345257630335, 1.033038655033, 1.562198315728, 1.350165690206};

    //// Moller
    EXPECT_VEC_SOFT_EQ(expected_m_inc_exit_cost, m_results.inc_exit_cost);
    EXPECT_VEC_SOFT_EQ(expected_m_inc_exit_e, m_results.inc_exit_e);
    EXPECT_VEC_SOFT_EQ(expected_m_inc_edep, m_results.inc_edep);
    EXPECT_VEC_SOFT_EQ(expected_m_sec_cost, m_results.sec_cost);
    EXPECT_VEC_SOFT_EQ(expected_m_sec_e, m_results.sec_e);
    for (const auto secondary_energy : m_results.sec_e)
    {
        // Verify if secondary is above the cutoff threshold
        EXPECT_TRUE(secondary_energy > cutoff_view.energy().value());
    }

    //// Bhabha
    EXPECT_VEC_SOFT_EQ(expected_b_inc_exit_cost, b_results.inc_exit_cost);
    EXPECT_VEC_SOFT_EQ(expected_b_inc_exit_e, b_results.inc_exit_e);
    EXPECT_VEC_SOFT_EQ(expected_b_inc_edep, b_results.inc_edep);
    EXPECT_VEC_SOFT_EQ(expected_b_sec_cost, b_results.sec_cost);
    EXPECT_VEC_SOFT_EQ(expected_b_sec_e, b_results.sec_e);
    for (const auto secondary_energy : b_results.sec_e)
    {
        // Verify if secondary is above the cutoff threshold
        EXPECT_TRUE(secondary_energy > cutoff_view.energy().value());
    }
}

//---------------------------------------------------------------------------//
TEST_F(MollerBhabhaInteractorTest, stress_test)
{
    RandomEngine& rng = this->rng();

    const int           num_samples = 1e4;
    std::vector<double> avg_engine_samples;

    CutoffView cutoff_view(
        this->cutoff_params()->host_pointers(), ParticleId{0}, MaterialId{0});

    // Moller's max energy fraction is 0.5, which leads to E_K > 2e-3
    // Bhabha's max energy fraction is 1.0, which leads to E_K > 1e-3
    // Since this loop encompasses both Moller and Bhabha, the minimum chosen
    // energy is > 2e-3.
    // NOTE: As E_K -> 2e-3, engine_samples -> infinity
    for (auto particle : {pdg::electron(), pdg::positron()})
    {
        for (double inc_e : {5e-3, 1.0, 10.0, 100.0, 1000.0})
        {
            RandomEngine::size_type num_particles_sampled = 0;

            // Loop over several incident directions (shouldn't affect anything
            // substantial, but scattering near Z axis loses precision)
            for (const Real3& inc_dir : {Real3{0, 0, 1},
                                         Real3{1, 0, 0},
                                         Real3{1e-9, 0, 1},
                                         Real3{1, 1, 1}})
            {
                this->set_inc_direction(inc_dir);
                this->resize_secondaries(num_samples);

                // Create interactor
                this->set_inc_particle(particle, MevEnergy{inc_e});
                MollerBhabhaInteractor mb_interact(pointers_,
                                                   this->particle_track(),
                                                   cutoff_view,
                                                   this->direction(),
                                                   this->secondary_allocator());

                // Loop over half the sample size
                for (int i = 0; i < num_samples; ++i)
                {
                    Interaction result = mb_interact(rng);
                    this->sanity_check(result);
                }

                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(double(rng.count())
                                         / double(num_particles_sampled));
            rng.reset_count();
        }
    }
    // Gold values for average number of calls to rng
    const double expected_avg_engine_samples[] = {29.6929,
                                                  13.2098,
                                                  9.5096,
                                                  9.1994,
                                                  9.1725,
                                                  564.227,
                                                  8.7404,
                                                  7.1755,
                                                  7.0203,
                                                  6.9907};

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}
