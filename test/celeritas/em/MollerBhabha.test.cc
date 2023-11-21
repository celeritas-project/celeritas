//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MollerBhabha.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/MollerBhabhaInteractor.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/InteractionIO.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using units::MevEnergy;

class MollerBhabhaInteractorTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using namespace units;

        // Setup MaterialView
        MaterialParams::Input inp;
        inp.elements = {{AtomicNumber{29}, units::AmuMass{63.546}, {}, "Cu"}};
        inp.materials = {
            {1.0 * constants::na_avogadro,
             293.0,
             MatterState::solid,
             {{ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(inp);
        this->set_material("Cu");

        // Set 1 keV cutoffs
        CutoffParams::Input cutoff_inp;
        cutoff_inp.materials = this->material_params();
        cutoff_inp.particles = this->particle_params();
        cutoff_inp.cutoffs = {{pdg::electron(), {{MevEnergy{0.001}, 0.1234}}}};
        this->set_cutoff_params(cutoff_inp);

        // Set MollerBhabhaData
        auto const& params = *this->particle_params();
        data_.ids.electron = params.find(pdg::electron());
        data_.ids.positron = params.find(pdg::positron());
        data_.electron_mass = params.get(data_.ids.electron).mass();
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(interaction.direction));
        EXPECT_EQ(Action::scattered, interaction.action);

        // Check secondaries
        ASSERT_EQ(1, interaction.secondaries.size());
        auto const& electron = interaction.secondaries.front();
        EXPECT_TRUE(electron);
        EXPECT_EQ(data_.ids.electron, electron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  electron.energy.value());
        EXPECT_LT(0, electron.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(electron.direction));

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
    }

  protected:
    MollerBhabhaData data_;
};

struct SampleInit
{
    real_type energy;  //!< MeV
    Real3 dir;
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

    CutoffView cutoff_view(this->cutoff_params()->host_ref(), MaterialId{0});

    for (SampleInit const& init : samples)
    {
        Real3 dir = init.dir;
        normalize_direction(&dir);
        this->set_inc_direction(dir);

        for (auto p : {pdg::electron(), pdg::positron()})
        {
            this->set_inc_particle(p, MevEnergy{init.energy});

            MollerBhabhaInteractor mb_interact(data_,
                                               this->particle_track(),
                                               cutoff_view,
                                               dir,
                                               this->secondary_allocator());

            Interaction result = mb_interact(rng_engine);
            this->sanity_check(result);
            Secondary const& sec = result.secondaries.front();

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
    double const expected_m_inc_exit_cost[]
        = {0.9981497250995, 0.999993612333, 0.999999995461, 0.9999999999998};
    double const expected_m_inc_exit_e[]
        = {0.9927116916645, 9.998622388005, 999.9911084469, 99999.99528134};
    double const expected_m_inc_edep[] = {0, 0, 0, 0};
    double const expected_m_sec_cost[] = {
        0.1196563201983, 0.03851909820188, 0.09291901073767, 0.06779325842364};
    double const expected_m_sec_e[] = {0.007288308335526,
                                       0.001377611995461,
                                       0.008891553104294,
                                       0.004718664979811};

    //// Bhabha
    // Gold values based on the host rng. Energies are in MeV
    double const expected_b_inc_exit_cost[]
        = {0.9997453107903, 0.999994190499, 0.9999999989865, 0.9999999999999};
    double const expected_b_inc_exit_e[]
        = {0.9989928374838, 9.998747065072, 999.9980145461, 99999.99864983};
    double const expected_b_inc_edep[] = {0, 0, 0, 0};
    double const expected_b_sec_cost[] = {
        0.04461708949452, 0.03673697288345, 0.04405602044159, 0.03632326062515};
    double const expected_b_sec_e[] = {0.001007162516187,
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
    cutoff_inp.cutoffs.insert({pdg::positron(), material_cutoffs});
    this->set_cutoff_params(cutoff_inp);

    CutoffView cutoff_view(this->cutoff_params()->host_ref(), MaterialId{0});

    for (SampleInit const& init : samples)
    {
        Real3 dir = init.dir;
        normalize_direction(&dir);
        this->set_inc_direction(dir);

        for (auto p : {pdg::electron(), pdg::positron()})
        {
            this->set_inc_particle(p, MevEnergy{init.energy});

            MollerBhabhaInteractor mb_interact(data_,
                                               this->particle_track(),
                                               cutoff_view,
                                               dir,
                                               this->secondary_allocator());

            Interaction result = mb_interact(rng_engine);
            this->sanity_check(result);
            Secondary const& sec = result.secondaries.front();

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
    double const expected_m_inc_exit_cost[]
        = {0.9784675127353, 0.9997401875592, 0.9999953862586, 0.9999999997589};
    double const expected_m_inc_exit_e[]
        = {6.75726441249, 95.11275692125, 991.0427997072, 99995.28168559};
    double const expected_m_inc_edep[] = {0, 0, 0, 0};
    double const expected_m_sec_cost[]
        = {0.9154612855963, 0.91405872098, 0.9478947756541, 0.9066254320384};
    double const expected_m_sec_e[]
        = {3.24273558751, 4.887243078746, 8.957200292789, 4.718314414109};

    //// Bhabha
    // Gold values based on the host rng. Energies are in MeV
    double const expected_b_inc_exit_cost[]
        = {0.9774788335858, 0.9999472046111, 0.9999992012865, 0.999999999931};
    double const expected_b_inc_exit_e[]
        = {6.654742369665, 98.96696134497, 998.4378016843, 99998.64983431};
    double const expected_b_inc_edep[] = {0, 0, 0, 0};
    double const expected_b_sec_cost[]
        = {0.9188415916986, 0.7126175077086, 0.777906053136, 0.7544377929863};
    double const expected_b_sec_e[]
        = {3.345257630335, 1.033038655033, 1.562198315728, 1.350165690206};

    //// Moller
    EXPECT_VEC_SOFT_EQ(expected_m_inc_exit_cost, m_results.inc_exit_cost);
    EXPECT_VEC_SOFT_EQ(expected_m_inc_exit_e, m_results.inc_exit_e);
    EXPECT_VEC_SOFT_EQ(expected_m_inc_edep, m_results.inc_edep);
    EXPECT_VEC_SOFT_EQ(expected_m_sec_cost, m_results.sec_cost);
    EXPECT_VEC_SOFT_EQ(expected_m_sec_e, m_results.sec_e);
    for (auto const secondary_energy : m_results.sec_e)
    {
        // Verify if secondary is above the cutoff threshold
        EXPECT_TRUE(secondary_energy
                    > cutoff_view.energy(ParticleId{0}).value());
    }

    //// Bhabha
    EXPECT_VEC_SOFT_EQ(expected_b_inc_exit_cost, b_results.inc_exit_cost);
    EXPECT_VEC_SOFT_EQ(expected_b_inc_exit_e, b_results.inc_exit_e);
    EXPECT_VEC_SOFT_EQ(expected_b_inc_edep, b_results.inc_edep);
    EXPECT_VEC_SOFT_EQ(expected_b_sec_cost, b_results.sec_cost);
    EXPECT_VEC_SOFT_EQ(expected_b_sec_e, b_results.sec_e);
    for (auto const secondary_energy : b_results.sec_e)
    {
        // Verify if secondary is above the cutoff threshold
        EXPECT_TRUE(secondary_energy
                    > cutoff_view.energy(ParticleId{0}).value());
    }
}

//---------------------------------------------------------------------------//
TEST_F(MollerBhabhaInteractorTest, stress_test)
{
    int const num_samples = 1e4;
    std::vector<double> avg_engine_samples;

    CutoffView cutoff_view(this->cutoff_params()->host_ref(), MaterialId{0});

    // Moller's max energy fraction is 0.5, which leads to E_K > 2e-3
    // Bhabha's max energy fraction is 1.0, which leads to E_K > 1e-3
    // Since this loop encompasses both Moller and Bhabha, the minimum chosen
    // energy is > 2e-3.
    // NOTE: As E_K -> 2e-3, engine_samples -> infinity
    for (auto particle : {pdg::electron(), pdg::positron()})
    {
        ParticleParams const& pp = *this->particle_params();
        SCOPED_TRACE(pp.id_to_label(pp.find(particle)));
        for (real_type inc_e : {5e-3, 1.0, 10.0, 100.0, 1000.0})
        {
            RandomEngine& rng_engine = this->rng();
            RandomEngine::size_type num_particles_sampled = 0;

            // Loop over several incident directions (shouldn't affect anything
            // substantial, but scattering near Z axis loses precision)
            for (Real3 const& inc_dir : {Real3{0, 0, 1},
                                         Real3{1, 0, 0},
                                         Real3{1e-9, 0, 1},
                                         Real3{1, 1, 1}})
            {
                this->set_inc_direction(inc_dir);
                this->resize_secondaries(num_samples);

                // Create interactor
                this->set_inc_particle(particle, MevEnergy{inc_e});
                MollerBhabhaInteractor mb_interact(data_,
                                                   this->particle_track(),
                                                   cutoff_view,
                                                   this->direction(),
                                                   this->secondary_allocator());

                // Loop over half the sample size
                for (int i = 0; i < num_samples; ++i)
                {
                    Interaction result = mb_interact(rng_engine);
                    this->sanity_check(result);
                    if (this->HasFailure())
                    {
                        // Only do one comparison in case of failure
                        break;
                    }
                }

                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(double(rng_engine.count())
                                         / double(num_particles_sampled));
        }
    }

    // Gold values for average number of calls to rng
    double const expected_avg_engine_samples[] = {20.8046,
                                                  13.2538,
                                                  9.5695,
                                                  9.2121,
                                                  9.1693,
                                                  564.0656,
                                                  8.7123,
                                                  7.1706,
                                                  7.0299,
                                                  7.0079};

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
