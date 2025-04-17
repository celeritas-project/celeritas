//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/LivermorePE.test.cc
//---------------------------------------------------------------------------//
#include <cmath>
#include <fstream>
#include <map>

#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/detail/Utils.hh"
#include "celeritas/em/interactor/LivermorePEInteractor.hh"
#include "celeritas/em/model/LivermorePEModel.hh"
#include "celeritas/em/params/AtomicRelaxationParams.hh"
#include "celeritas/em/xs/LivermorePEMicroXsCalculator.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/io/AtomicRelaxationReader.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/io/LivermorePEReader.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/InteractionIO.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"
#include "celeritas/phys/MacroXsCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class LivermorePETest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    using MevEnergy = units::MevEnergy;

    void set_relaxation_params(AtomicRelaxationParams::Input inp)
    {
        relax_params_
            = std::make_shared<AtomicRelaxationParams>(std::move(inp));
        relax_params_ref_ = relax_params_->host_ref();
    }

    void SetUp() override
    {
        using namespace units;
        using namespace constants;

        // Set up shared particle data
        auto const& particles = *this->particle_params();

        // Set up shared material data
        MaterialParams::Input mi;
        mi.elements = {{AtomicNumber{19}, AmuMass{39.0983}, {}, "K"}};
        mi.materials = {{native_value_from(MolCcDensity{1e-5}),
                         293.,
                         MatterState::solid,
                         {{ElementId{0}, 1.0}},
                         "K"}};
        this->set_material_params(mi);

        // Set cutoffs (no cuts)
        CutoffParams::Input ci;
        ci.materials = this->material_params();
        ci.particles = this->particle_params();
        this->set_cutoff_params(ci);

        // Set Livermore photoelectric data
        std::string data_path = this->test_data_path("celeritas", "");
        LivermorePEReader read_element_data(data_path.c_str());
        model_ = std::make_shared<LivermorePEModel>(
            ActionId{0}, particles, *this->material_params(), read_element_data);

        // Set atomic relaxation data
        AtomicRelaxationReader read_transition_data(data_path.c_str(),
                                                    data_path.c_str());
        relax_inp_.cutoffs = this->cutoff_params();
        relax_inp_.materials = this->material_params();
        relax_inp_.particles = this->particle_params();
        relax_inp_.load_data = read_transition_data;

        // Set default particle to incident 1 keV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{0.001});
        this->set_inc_direction({0, 0, 1});
        this->set_material("K");
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_EQ(Action::absorbed, interaction.action);

        // Check secondaries
        ASSERT_GT(2, interaction.secondaries.size());
        if (interaction.secondaries.size() == 1)
        {
            auto const& electron = interaction.secondaries.front();
            EXPECT_TRUE(electron);
            EXPECT_EQ(model_->host_ref().ids.electron, electron.particle_id);
            EXPECT_GE(this->particle_track().energy().value(),
                      electron.energy.value());
            EXPECT_LT(0, electron.energy.value());
            EXPECT_SOFT_EQ(1.0, norm(electron.direction));
        }

        // Check conservation between primary and secondaries. Since momentum
        // is transferred to the atom, we don't expect it to be conserved
        // between the incoming and outgoing particles
        this->check_energy_conservation(interaction);
    }

  protected:
    std::shared_ptr<LivermorePEModel> model_;
    AtomicRelaxationParams::Input relax_inp_;
    std::shared_ptr<AtomicRelaxationParams> relax_params_;
    HostVal<AtomicRelaxStateData> relax_states_;
    HostCRef<AtomicRelaxParamsData> relax_params_ref_;
    HostRef<AtomicRelaxStateData> relax_states_ref_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LivermorePETest, basic)
{
    RandomEngine& rng_engine = this->rng();

    // Reserve 4 secondaries
    this->resize_secondaries(4);

    // Sampled element
    ElementId el_id{0};

    // Production cuts
    auto cutoffs = this->cutoff_params()->get(PhysMatId{0});

    // Helper for simulating atomic relaxation
    AtomicRelaxationHelper relaxation(
        relax_params_ref_, relax_states_ref_, el_id, TrackSlotId{0});
    EXPECT_FALSE(relaxation);

    // Create the interactor
    LivermorePEInteractor interact(model_->host_ref(),
                                   relaxation,
                                   el_id,
                                   this->particle_track(),
                                   cutoffs,
                                   this->direction(),
                                   this->secondary_allocator());

    std::vector<real_type> energy_electron;
    std::vector<real_type> costheta_electron;
    std::vector<real_type> energy_deposition;

    // Produce four samples from the original incident energy/dir
    for (int i : range(4))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);
        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data() + i);

        // Add actual results to vector
        energy_electron.push_back(result.secondaries.front().energy.value());
        costheta_electron.push_back(dot_product(
            result.secondaries.front().direction, this->direction()));
        energy_deposition.push_back(result.energy_deposition.value());
    }

    EXPECT_EQ(4, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    real_type const expected_energy_electron[]
        = {0.00062884, 0.00062884, 0.00070136, 0.00069835};
    real_type const expected_costheta_electron[] = {
        0.1217302869581, 0.8769397871407, -0.1414717733267, -0.2414106440617};
    real_type const expected_energy_deposition[]
        = {0.00037116, 0.00037116, 0.00029864, 0.00030165};
    EXPECT_VEC_SOFT_EQ(expected_energy_electron, energy_electron);
    EXPECT_VEC_SOFT_EQ(expected_costheta_electron, costheta_electron);
    EXPECT_VEC_SOFT_EQ(expected_energy_deposition, energy_deposition);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(LivermorePETest, stress_test)
{
    int const num_samples = 8192;
    std::vector<real_type> avg_engine_samples;
    std::vector<real_type> avg_num_secondaries;
    std::vector<real_type> avg_cosine;
    std::vector<real_type> avg_energy;

    // Sampled element
    ElementId el_id{0};

    // Production cuts
    auto cutoffs = this->cutoff_params()->get(PhysMatId{0});

    // Helper for simulating atomic relaxation
    AtomicRelaxationHelper relaxation(
        relax_params_ref_, relax_states_ref_, el_id, TrackSlotId{0});
    EXPECT_FALSE(relaxation);

    for (real_type inc_e : {0.0001, 0.01, 1.0, 10.0, 1000.0})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

        RandomEngine& rng_engine = this->rng();
        RandomEngine::size_type num_particles_sampled = 0;
        RandomEngine::size_type num_secondaries = 0;
        real_type tot_cosine = 0;
        real_type tot_energy = 0;

        // Loop over several incident directions
        for (Real3 const& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(num_samples);

            // Create interactor
            LivermorePEInteractor interact(model_->host_ref(),
                                           relaxation,
                                           el_id,
                                           this->particle_track(),
                                           cutoffs,
                                           this->direction(),
                                           this->secondary_allocator());

            // Loop over many particles
            for (int i = 0; i < num_samples; ++i)
            {
                Interaction result = interact(rng_engine);
                // SCOPED_TRACE(result);
                this->sanity_check(result);
                for (auto const& sec : result.secondaries)
                {
                    tot_cosine += dot_product(inc_dir, sec.direction);
                    tot_energy += sec.energy.value();
                }
                num_secondaries += result.secondaries.size();
            }
            EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(real_type(rng_engine.count())
                                     / real_type(num_particles_sampled));
        avg_num_secondaries.push_back(real_type(num_secondaries)
                                      / real_type(num_particles_sampled));
        avg_cosine.push_back(tot_cosine / real_type(num_secondaries));
        avg_energy.push_back(tot_energy / real_type(num_secondaries));
    }

    // Gold values
    real_type const expected_avg_engine_samples[]
        = {15.99755859375, 16.09204101562, 13.79919433594, 8.590209960938, 2};
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);

    real_type const expected_avg_num_secondaries[] = {1, 1, 1, 1, 1};
    EXPECT_VEC_SOFT_EQ(expected_avg_num_secondaries, avg_num_secondaries);

    real_type const expected_avg_cosine[] = {0.0181237765392,
                                             0.1848443587223,
                                             1.030717821907,
                                             1.169482513617,
                                             1.183012701892};
    EXPECT_VEC_SOFT_EQ(expected_avg_cosine, expected_avg_cosine);

    real_type const expected_avg_energy[] = {7.287875885011e-05,
                                             0.006708485731503,
                                             0.9967066970311,
                                             9.996704339284,
                                             999.9967069717};
    EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
}

TEST_F(LivermorePETest, distributions_all)
{
    RandomEngine& rng_engine = this->rng();

    int const num_samples = 1000;
    Real3 inc_direction = {0, 0, 1};
    this->set_inc_direction(inc_direction);

    // Sampled element
    ElementId el_id{0};

    // Production cuts
    auto cutoffs = this->cutoff_params()->get(PhysMatId{0});

    // Load atomic relaxation data
    relax_inp_.is_auger_enabled = true;
    this->set_relaxation_params(relax_inp_);
    EXPECT_EQ(3, relax_params_ref_.max_stack_size);

    // Allocate scratch space for vacancy stack
    resize(&relax_states_, relax_params_ref_, 1);
    relax_states_ref_ = relax_states_;
    EXPECT_EQ(1, relax_states_ref_.size());

    // Helper for simulating atomic relaxation
    AtomicRelaxationHelper relaxation(
        relax_params_ref_, relax_states_ref_, el_id, TrackSlotId{0});
    EXPECT_EQ(7, relaxation.max_secondaries());

    // Allocate storage for secondaries (atomic relaxation + photoelectron)
    int secondary_size = (relaxation.max_secondaries() + 1) * num_samples;
    this->resize_secondaries(secondary_size);

    // Create the interactor
    LivermorePEInteractor interact(model_->host_ref(),
                                   relaxation,
                                   el_id,
                                   this->particle_track(),
                                   cutoffs,
                                   this->direction(),
                                   this->secondary_allocator());

    int nbins = 10;
    int num_secondaries = 0;
    std::map<real_type, int> energy_to_count;
    std::vector<real_type> energy;
    std::vector<int> count;
    std::vector<real_type> costheta_dist(nbins);

    // Loop over many particles
    for (int i = 0; i < num_samples; ++i)
    {
        Interaction out = interact(rng_engine);
        // SCOPED_TRACE(out);
        this->check_energy_conservation(out);
        num_secondaries += out.secondaries.size();

        // Bin directional change of the photoelectron
        real_type costheta
            = dot_product(inc_direction, out.secondaries.front().direction);
        int ct_bin = static_cast<int>(std::floor((1 + costheta) / 2 * nbins));
        if (ct_bin >= 0 && ct_bin < nbins)
        {
            ++costheta_dist[ct_bin];
        }

        for (auto const& secondary : out.secondaries)
        {
            // Increment the count of the discrete sampled energy
            energy_to_count[secondary.energy.value()]++;
        }
    }
    EXPECT_EQ(secondary_size, this->secondary_allocator().get().size());
    EXPECT_EQ(2180, num_secondaries);

    for (auto const& it : energy_to_count)
    {
        energy.push_back(it.first);
        count.push_back(it.second);
    }
    real_type const expected_costheta_dist[]
        = {23, 61, 83, 129, 135, 150, 173, 134, 85, 27};
    real_type const expected_energy[] = {
        2.901e-05,  3.202e-05,  4.576e-05,  4.604e-05,  4.877e-05,  4.905e-05,
        6.529e-05,  6.83e-05,   0.00021764, 0.00022065, 0.00023439, 0.00023467,
        0.0002374,  0.00023768, 0.00025114, 0.00025142, 0.0002517,  0.00025392,
        0.00025415, 0.00025443, 0.00025471, 0.00027095, 0.00027368, 0.00029016,
        0.00030691, 0.00030719, 0.00034347, 0.00062884, 0.00069835, 0.00070136,
        0.0009595,  0.00097625, 0.00097653,
    };
    int const expected_count[] = {
        42, 80, 26,  24, 27, 54, 2, 5, 5,  5, 4,   141, 61,  3,  2,  169, 260,
        1,  39, 195, 2,  8,  5,  3, 2, 14, 1, 280, 216, 424, 32, 16, 32};
    EXPECT_VEC_EQ(expected_costheta_dist, costheta_dist);
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_EQ(expected_count, count);
}

TEST_F(LivermorePETest, distributions_radiative)
{
    RandomEngine& rng_engine = this->rng();

    int const num_samples = 10000;

    // Sampled element
    ElementId el_id{0};

    // Production cuts
    auto cutoffs = this->cutoff_params()->get(PhysMatId{0});

    // Load atomic relaxation data
    relax_inp_.is_auger_enabled = false;
    this->set_relaxation_params(relax_inp_);
    EXPECT_EQ(1, relax_params_ref_.max_stack_size);

    // Allocate scratch space for vacancy stack
    resize(&relax_states_, relax_params_ref_, 1);
    relax_states_ref_ = relax_states_;
    EXPECT_EQ(1, relax_states_ref_.size());

    // Helper for simulating atomic relaxation
    AtomicRelaxationHelper relaxation(
        relax_params_ref_, relax_states_ref_, el_id, TrackSlotId{0});
    EXPECT_EQ(3, relaxation.max_secondaries());

    // Allocate storage for secondaries (atomic relaxation + photoelectron)
    int secondary_size = (relaxation.max_secondaries() + 1) * num_samples;
    this->resize_secondaries(secondary_size);

    // Create the interactor
    LivermorePEInteractor interact(model_->host_ref(),
                                   relaxation,
                                   el_id,
                                   this->particle_track(),
                                   cutoffs,
                                   this->direction(),
                                   this->secondary_allocator());

    int num_secondaries = 0;
    std::map<real_type, int> energy_to_count;
    std::vector<real_type> energy;
    std::vector<int> count;

    // Loop over many particles
    for (int i = 0; i < num_samples; ++i)
    {
        Interaction out = interact(rng_engine);
        // SCOPED_TRACE(out);
        this->check_energy_conservation(out);
        num_secondaries += out.secondaries.size();

        for (auto const& secondary : out.secondaries)
        {
            // Increment the count of the discrete sampled energy
            energy_to_count[secondary.energy.value()]++;
        }
    }
    EXPECT_EQ(secondary_size, this->secondary_allocator().get().size());
    EXPECT_EQ(10007, num_secondaries);

    for (auto const& it : energy_to_count)
    {
        energy.push_back(it.first);
        count.push_back(it.second);
    }
    real_type const expected_energy[] = {
        6.951e-05,
        0.00025814,
        0.00026115,
        0.00034741,
        0.00034769,
        0.00062884,
        0.00069835,
        0.00070136,
        0.0009595,
        0.00097625,
        0.00097653,
        0.00099578,
    };
    int const expected_count[]
        = {2, 1, 1, 1, 2, 2525, 2228, 4358, 337, 181, 361, 10};
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_EQ(expected_count, count);
}

TEST_F(LivermorePETest, macro_xs)
{
    auto material = this->material_track().material_record();
    auto calc_macro_xs = MacroXsCalculator<LivermorePEMicroXsCalculator>(
        model_->host_ref(), material);

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
        = {9.235615290944,     17.56658325086,     1.161217594282,
           0.4108511065363,    0.01515608909912,   0.0004000659204694,
           9.083754758322e-06, 2.449452106704e-07, 1.800625084911e-08,
           3.188458732396e-09, 8.028833591133e-10, 2.2700912115e-10,
           6.653075041804e-11, 1.971081007251e-11, 5.85857761177e-12,
           1.743005702864e-12, 5.187166124179e-13, 1.543827005416e-13,
           4.594922185898e-14, 1.367605938008e-14};
    EXPECT_VEC_SOFT_EQ(expected_macro_xs, macro_xs);
}
//---------------------------------------------------------------------------//
}  // namespace test

namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
// For an element with n shells of transition data, the maximum number of
// secondaries created can be upper-bounded as n if there are only
// radiative transitions and 2^n - 1 if there are non-radiative transitions
// for the hypothetical worst case where for a given vacancy the
// transitions always originate from the next subshell up
class LivermorePEUtilsTest : public ::celeritas::test::LivermorePETest
{
  public:
    using Values = AtomicRelaxParamsData<Ownership::value, MemSpace::host>;

    static unsigned int const num_shells;
};
unsigned int const LivermorePEUtilsTest::num_shells = 20;

/*!
 * One radiative transition per subshell, each one originating in the next
 * subshell up.
 */
TEST_F(LivermorePEUtilsTest, one_neighbor)
{
    Values data;
    resize(&data.elements, 1);
    AtomicRelaxElement& el = data.elements[ElementId(0)];

    std::vector<AtomicRelaxSubshell> shells(num_shells);
    for (auto i : range(num_shells))
    {
        std::vector<AtomicRelaxTransition> transitions
            = {{SubshellId{i + 1}, SubshellId{}, real_type{1}, MevEnergy{1}}};
        shells[i].transitions
            = make_builder(&data.transitions)
                  .insert_back(transitions.begin(), transitions.end());
    }
    el.shells
        = make_builder(&data.shells).insert_back(shells.begin(), shells.end());

    auto max_secondaries = calc_max_secondaries(
        make_const_ref(data), el.shells, MevEnergy{0}, MevEnergy{0});
    EXPECT_EQ(num_shells, max_secondaries);

    // If there are only radiative transitions, there will only ever be one
    // vacancy on the stack
    auto max_stack_size = calc_max_stack_size(make_const_ref(data), el.shells);
    EXPECT_EQ(1, max_stack_size);
}

/*!
 * num_shells - subshell_id non-radiative transitions per subshell, one
 * originating in each of the higher subshells
 */
TEST_F(LivermorePEUtilsTest, one_per_previous)
{
    Values data;
    resize(&data.elements, 1);
    AtomicRelaxElement& el = data.elements[ElementId(0)];

    std::vector<AtomicRelaxSubshell> shells(num_shells);
    for (auto i : range(num_shells))
    {
        std::vector<AtomicRelaxTransition> transitions;
        for (auto j : range(i, num_shells))
        {
            transitions.push_back({SubshellId{j + 1},
                                   SubshellId{j + 1},
                                   real_type{1} / (num_shells - i),
                                   MevEnergy{1}});
        }
        shells[i].transitions
            = make_builder(&data.transitions)
                  .insert_back(transitions.begin(), transitions.end());
    }
    el.shells
        = make_builder(&data.shells).insert_back(shells.begin(), shells.end());

    auto max_secondaries = calc_max_secondaries(
        make_const_ref(data), el.shells, MevEnergy{0}, MevEnergy{0});
    EXPECT_EQ(std::exp2(num_shells) - 1, max_secondaries);

    // With non-radiative transitions in every shell, the maximum stack
    // size will be one more than the number of shells with transition data
    auto max_stack_size = calc_max_stack_size(make_const_ref(data), el.shells);
    EXPECT_EQ(num_shells + 1, max_stack_size);
}

TEST_F(LivermorePEUtilsTest, auger)
{
    relax_inp_.is_auger_enabled = true;
    CutoffParams::Input ci;
    ci.materials = this->material_params();
    ci.particles = this->particle_params();

    // Test 1 keV electron/photon cutoff
    ci.cutoffs[pdg::electron()] = {{MevEnergy{1.e-3}, 0}};
    ci.cutoffs[pdg::gamma()] = {{MevEnergy{1.e-3}, 0}};
    this->set_cutoff_params(ci);

    ElementId el{0};
    relax_inp_.cutoffs = this->cutoff_params();
    this->set_relaxation_params(relax_inp_);
    EXPECT_EQ(1, relax_params_ref_.elements[el].max_secondary);

    // Test 0.1 keV electron/photon cutoff
    ci.cutoffs[pdg::electron()] = {{MevEnergy{1.e-4}, 0}};
    ci.cutoffs[pdg::gamma()] = {{MevEnergy{1.e-4}, 0}};
    this->set_cutoff_params(ci);

    relax_inp_.cutoffs = this->cutoff_params();
    this->set_relaxation_params(relax_inp_);
    EXPECT_EQ(3, relax_params_ref_.elements[el].max_secondary);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
