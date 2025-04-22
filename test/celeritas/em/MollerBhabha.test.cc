//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
            {native_value_from(MolCcDensity{1.0}),
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

    CutoffView cutoff_view(this->cutoff_params()->host_ref(), PhysMatId{0});

    for (SampleInit const& init : samples)
    {
        Real3 dir = make_unit_vector(init.dir);
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
    static double const expected_m_inc_exit_cost[] = {
        0.99973900914319, 0.99998439403408, 0.99999999892476, 0.99999999999959};
    static double const expected_m_inc_exit_e[]
        = {0.99896793373796, 9.996634923266, 999.9978936714, 99999.992056977};
    static double const expected_m_inc_edep[] = {0, 0, 0, 0};
    static double const expected_m_sec_cost[] = {0.045164786751542,
                                                 0.060143518761038,
                                                 0.045374598488427,
                                                 0.087819098576717};
    static double const expected_m_sec_e[] = {0.0010320662620386,
                                              0.0033650767340367,
                                              0.0021063286005393,
                                              0.0079430230464576};

    //// Bhabha
    // Gold values based on the host rng. Energies are in MeV
    static double const expected_b_inc_exit_cost[] = {
        0.99953803363407, 0.9999871668284, 0.99999999937891, 0.99999999999991};
    static double const expected_b_inc_exit_e[] = {
        0.99817409418782, 9.9972326602584, 999.99878331207, 99999.998278701};
    static double const expected_b_inc_edep[] = {0, 0, 0, 0};
    static double const expected_b_sec_cost[] = {0.060050541049384,
                                                 0.054556840397858,
                                                 0.034500711450312,
                                                 0.041005293173408};
    static double const expected_b_sec_e[] = {0.0018259058121848,
                                              0.0027673397416426,
                                              0.0012166879316949,
                                              0.001721298732305};

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

    CutoffView cutoff_view(this->cutoff_params()->host_ref(), PhysMatId{0});

    for (SampleInit const& init : samples)
    {
        Real3 dir = make_unit_vector(init.dir);
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
    static double const expected_m_inc_exit_cost[] = {
        0.99474381994671, 0.9998320422927, 0.99999935924924, 0.99999999959414};
    static double const expected_m_inc_exit_e[]
        = {8.9744580752619, 96.785484384822, 998.74637287344, 99992.05807867};
    static double const expected_m_inc_edep[] = {0, 0, 0, 0};
    static double const expected_m_sec_cost[] = {
        0.74300321590697, 0.87551068112013, 0.74260120785797, 0.94127395616289};
    static double const expected_m_sec_e[]
        = {1.0255419247381, 3.2145156151784, 1.2536271265614, 7.9419213303979};

    //// Bhabha
    // Gold values based on the host rng. Energies are in MeV
    static double const expected_b_inc_exit_cost[] = {
        0.99479050564194, 0.99987897158914, 0.99999937828724, 0.99999999991204};
    static double const expected_b_inc_exit_e[]
        = {8.9827046800673, 97.662824061219, 998.78357538952, 99998.278713671};
    static double const expected_b_inc_edep[] = {0, 0, 0, 0};
    static double const expected_b_sec_cost[] = {
        0.74150459775358, 0.83837330389592, 0.73755324517749, 0.79212437269015};
    static double const expected_b_sec_e[]
        = {1.0172953199327, 2.3371759387812, 1.2164246104832, 1.721286329104};

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
    int const num_samples = 10000;
    std::vector<double> avg_engine_samples;

    CutoffView cutoff_view(this->cutoff_params()->host_ref(), PhysMatId{0});

    // Moller's max energy fraction is 0.5, which leads to E_K > 2e-3
    // Bhabha's max energy fraction is 1.0, which leads to E_K > 1e-3
    // Since this loop encompasses both Moller and Bhabha, the minimum chosen
    // energy is > 2e-3.
    for (auto particle : {pdg::electron(), pdg::positron()})
    {
        ParticleParams const& pp = *this->particle_params();
        SCOPED_TRACE(pp.id_to_label(pp.find(particle)));
        for (real_type inc_e : {2.0001e-3, 5e-3, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5})
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
    static double const expected_avg_engine_samples[] = {6,
                                                         7.093,
                                                         8.2035,
                                                         10.5552,
                                                         10.9619,
                                                         10.9763,
                                                         11.0187,
                                                         11.0541,
                                                         6.0111,
                                                         6.0269,
                                                         11.8788,
                                                         19.8378,
                                                         21.7165,
                                                         21.8738,
                                                         21.8782,
                                                         22.3027};

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
