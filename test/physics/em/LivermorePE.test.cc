//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePE.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/detail/LivermorePEInteractor.hh"

#include <cmath>
#include <fstream>
#include <map>
#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "comm/Device.hh"
#include "io/AtomicRelaxationReader.hh"
#include "io/ImportPhysicsTable.hh"
#include "io/LivermorePEParamsReader.hh"
#include "physics/base/Units.hh"
#include "physics/em/AtomicRelaxationParams.hh"
#include "physics/em/LivermorePEModel.hh"
#include "physics/em/LivermorePEParams.hh"
#include "physics/em/PhotoelectricProcess.hh"
#include "physics/em/LivermorePEMacroXsCalculator.hh"
#include "physics/grid/XsCalculator.hh"
#include "physics/grid/ValueGridBuilder.hh"
#include "physics/grid/ValueGridInserter.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/em/detail/Utils.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::Applicability;
using celeritas::AtomicRelaxationParams;
using celeritas::AtomicRelaxationReader;
using celeritas::ElementId;
using celeritas::ImportPhysicsTable;
using celeritas::ImportPhysicsVectorType;
using celeritas::ImportTableType;
using celeritas::LivermorePEMacroXsCalculator;
using celeritas::LivermorePEParams;
using celeritas::LivermorePEParamsReader;
using celeritas::MemSpace;
using celeritas::Ownership;
using celeritas::PhotoelectricProcess;
using celeritas::SubshellId;
using celeritas::ValueGridInserter;
using celeritas::detail::LivermorePEInteractor;
namespace pdg = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class LivermorePEInteractorTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void set_livermore_params(LivermorePEParams::Input inp)
    {
        CELER_EXPECT(!inp.elements.empty());
        livermore_params_ = std::make_shared<LivermorePEParams>(std::move(inp));
    }

    void set_relaxation_params(AtomicRelaxationParams::Input inp)
    {
        CELER_EXPECT(!inp.elements.empty());
        relax_params_
            = std::make_shared<AtomicRelaxationParams>(std::move(inp));
    }

    void SetUp() override
    {
        using celeritas::MatterState;
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        using namespace celeritas::constants;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // Set up shared particle data
        Base::set_particle_params(
            {{"electron",
              pdg::electron(),
              MevMass{0.5109989461},
              ElementaryCharge{-1},
              stable},
             {"gamma", pdg::gamma(), zero, zero, stable}});

        const auto& params    = this->particle_params();
        std::string data_path = this->test_data_path("physics/em", "");

        // Set Livermore photoelectric data
        LivermorePEParams::Input li;
        LivermorePEParamsReader read_element_data(data_path.c_str());
        li.elements.push_back(read_element_data(19));
        set_livermore_params(li);

        // Set atomic relaxation data
        AtomicRelaxationReader read_transition_data(data_path.c_str(),
                                                    data_path.c_str());
        relax_inp_.elements.push_back(read_transition_data(19));
        relax_inp_.electron_id = params.find(pdg::electron());
        relax_inp_.gamma_id    = params.find(pdg::gamma());

        // Set Livermore PE model interface
        pointers_.electron_id = params.find(pdg::electron());
        pointers_.gamma_id    = params.find(pdg::gamma());
        pointers_.inv_electron_mass
            = 1 / (params.get(pointers_.electron_id).mass().value());
        pointers_.data = livermore_params_->host_pointers();

        // Set default particle to incident 1 keV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{0.001});
        this->set_inc_direction({0, 0, 1});

        // Set up shared material data
        MaterialParams::Input mi;
        mi.elements  = {{19, AmuMass{39.0983}, "K"}};
        mi.materials = {{1e-5 * na_avogadro,
                         293.,
                         MatterState::solid,
                         {{ElementId{0}, 1.0}},
                         "K"}};

        // Set default material to potassium
        this->set_material_params(mi);
        this->set_material("K");
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_EQ(celeritas::Action::absorbed, interaction.action);

        // Check secondaries
        ASSERT_GT(2, interaction.secondaries.size());
        if (interaction.secondaries.size() == 1)
        {
            const auto& electron = interaction.secondaries.front();
            EXPECT_TRUE(electron);
            EXPECT_EQ(pointers_.electron_id, electron.particle_id);
            EXPECT_GT(this->particle_track().energy().value(),
                      electron.energy.value());
            EXPECT_LT(0, electron.energy.value());
            EXPECT_SOFT_EQ(1.0, celeritas::norm(electron.direction));
        }

        // Check conservation between primary and secondaries. Since momentum
        // is transferred to the atom, we don't expect it to be conserved
        // between the incoming and outgoing particles
        this->check_energy_conservation(interaction);
    }

    void resize_vacancies(int size)
    {
        CELER_EXPECT(size > 0);
        resize(&relax_store_.vacancies, size);
        scratch_ = relax_store_;
    }

  protected:
    AtomicRelaxationParams::Input           relax_inp_;
    std::shared_ptr<AtomicRelaxationParams> relax_params_;
    std::shared_ptr<LivermorePEParams>      livermore_params_;
    celeritas::detail::RelaxationScratchData<Ownership::value, MemSpace::host>
        relax_store_;

    celeritas::detail::LivermorePEPointers pointers_;
    celeritas::detail::RelaxationScratchData<Ownership::reference, MemSpace::host>
        scratch_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LivermorePEInteractorTest, basic)
{
    RandomEngine& rng_engine = this->rng();

    // Reserve 4 secondaries
    this->resize_secondaries(4);

    // Sampled element
    ElementId el_id{0};

    // Create the interactor
    LivermorePEInteractor interact(pointers_,
                                   scratch_,
                                   el_id,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());

    std::vector<double> energy_electron;
    std::vector<double> costheta_electron;
    std::vector<double> energy_deposition;

    // Produce four samples from the original incident energy/dir
    for (int i : celeritas::range(4))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);
        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data() + i);

        // Add actual results to vector
        energy_electron.push_back(result.secondaries.front().energy.value());
        costheta_electron.push_back(celeritas::dot_product(
            result.secondaries.front().direction, this->direction()));
        energy_deposition.push_back(result.energy_deposition.value());
    }

    EXPECT_EQ(4, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    const double expected_energy_electron[]
        = {0.00062884, 0.00062884, 0.00070136, 0.00069835};
    const double expected_costheta_electron[] = {
        0.1217302869581, 0.8769397871407, -0.1414717733267, -0.2414106440617};
    const double expected_energy_deposition[]
        = {0.00037116, 0.00037116, 0.00029864, 0.00030165};
    EXPECT_VEC_SOFT_EQ(expected_energy_electron, energy_electron);
    EXPECT_VEC_SOFT_EQ(expected_costheta_electron, costheta_electron);
    EXPECT_VEC_SOFT_EQ(expected_energy_deposition, energy_deposition);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}

TEST_F(LivermorePEInteractorTest, stress_test)
{
    RandomEngine& rng_engine = this->rng();

    const int           num_samples = 8192;
    std::vector<double> avg_engine_samples;

    ElementId el_id{0};

    for (double inc_e : {0.0001, 0.01, 1.0, 10.0, 1000.0})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});
        RandomEngine::size_type num_particles_sampled = 0;

        // Loop over several incident directions (shouldn't affect anything
        // substantial, but scattering near Z axis loses precision)
        for (const Real3& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(num_samples);

            // Create interactor
            LivermorePEInteractor interact(pointers_,
                                           scratch_,
                                           el_id,
                                           this->particle_track(),
                                           this->direction(),
                                           this->secondary_allocator());

            // Loop over many particles
            for (int i = 0; i < num_samples; ++i)
            {
                Interaction result = interact(rng_engine);
                SCOPED_TRACE(result);
                this->sanity_check(result);
            }
            EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(double(rng_engine.count())
                                     / double(num_particles_sampled));
        rng_engine.reset_count();
    }
    // PRINT_EXPECTED(avg_engine_samples);
    // Gold values for average number of calls to RNG
    const double expected_avg_engine_samples[]
        = {15.99755859375, 16.09204101562, 13.79919433594, 8.590209960938, 2};
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

TEST_F(LivermorePEInteractorTest, distributions_all)
{
    RandomEngine& rng_engine = this->rng();

    const int num_samples   = 1000;
    Real3     inc_direction = {0, 0, 1};
    this->set_inc_direction(inc_direction);

    // Sampled element
    ElementId el_id{0};

    // Add atomic relaxation data
    relax_inp_.is_auger_enabled = true;
    set_relaxation_params(relax_inp_);
    pointers_.atomic_relaxation = relax_params_->host_pointers();

    // Allocate space to hold unprocessed vacancy stack in atomic relaxation
    auto max_stack_size
        = pointers_.atomic_relaxation.elements[el_id.get()].max_stack_size;
    EXPECT_EQ(4, max_stack_size);
    this->resize_vacancies(max_stack_size * num_samples);

    // Allocate storage for secondaries (atomic relaxation + photoelectron)
    auto max_secondary
        = pointers_.atomic_relaxation.elements[el_id.get()].max_secondary + 1;
    EXPECT_EQ(8, max_secondary);
    this->resize_secondaries(max_secondary * num_samples);

    // Create the interactor
    LivermorePEInteractor interact(pointers_,
                                   scratch_,
                                   el_id,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());

    int                   nbins           = 10;
    int                   num_secondaries = 0;
    std::map<double, int> energy_to_count;
    std::vector<double>   energy;
    std::vector<int>      count;
    std::vector<double>   costheta_dist(nbins);

    // Loop over many particles
    for (int i = 0; i < num_samples; ++i)
    {
        Interaction out = interact(rng_engine);
        SCOPED_TRACE(out);
        ASSERT_TRUE(out);
        this->check_energy_conservation(out);
        num_secondaries += out.secondaries.size();

        // Bin directional change of the photoelectron
        double costheta = celeritas::dot_product(
            inc_direction, out.secondaries.front().direction);
        int ct_bin = (1 + costheta) / 2 * nbins; // Remap from [-1,1] to [0,1]
        if (ct_bin >= 0 && ct_bin < nbins)
        {
            ++costheta_dist[ct_bin];
        }

        for (const auto& secondary : out.secondaries)
        {
            // Increment the count of the discrete sampled energy
            energy_to_count[secondary.energy.value()]++;
        }
    }
    EXPECT_EQ(max_secondary * num_samples,
              this->secondary_allocator().get().size());
    EXPECT_EQ(2180, num_secondaries);

    for (const auto& it : energy_to_count)
    {
        energy.push_back(it.first);
        count.push_back(it.second);
    }
    const double expected_costheta_dist[]
        = {23, 61, 83, 129, 135, 150, 173, 134, 85, 27};
    const double expected_energy[] = {
        2.901e-05,  3.202e-05,  4.576e-05,  4.604e-05,  4.877e-05,  4.905e-05,
        6.529e-05,  6.83e-05,   0.00021764, 0.00022065, 0.00023439, 0.00023467,
        0.0002374,  0.00023768, 0.00025114, 0.00025142, 0.0002517,  0.00025392,
        0.00025415, 0.00025443, 0.00025471, 0.00027095, 0.00027368, 0.00029016,
        0.00030691, 0.00030719, 0.00034347, 0.00062884, 0.00069835, 0.00070136,
        0.0009595,  0.00097625, 0.00097653,
    };
    const int expected_count[] = {
        42, 80, 26,  24, 27, 54, 2, 5, 5,  5, 4,   141, 61,  3,  2,  169, 260,
        1,  39, 195, 2,  8,  5,  3, 2, 14, 1, 280, 216, 424, 32, 16, 32};
    EXPECT_VEC_EQ(expected_costheta_dist, costheta_dist);
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_EQ(expected_count, count);
}

TEST_F(LivermorePEInteractorTest, distributions_radiative)
{
    RandomEngine& rng_engine = this->rng();

    const int num_samples = 10000;

    // Sampled element
    ElementId el_id{0};

    // Add atomic relaxation data
    relax_inp_.is_auger_enabled = false;
    set_relaxation_params(relax_inp_);

    pointers_.atomic_relaxation = relax_params_->host_pointers();

    // Allocate space to hold unprocessed vacancy stack in atomic relaxation
    auto max_stack_size
        = pointers_.atomic_relaxation.elements[el_id.get()].max_stack_size;
    EXPECT_EQ(1, max_stack_size);
    this->resize_vacancies(max_stack_size * num_samples);

    // Allocate storage for secondaries (atomic relaxation + photoelectron)
    auto max_secondary
        = pointers_.atomic_relaxation.elements[el_id.get()].max_secondary + 1;
    EXPECT_EQ(4, max_secondary);
    this->resize_secondaries(max_secondary * num_samples);

    // Create the interactor
    LivermorePEInteractor interact(pointers_,
                                   scratch_,
                                   el_id,
                                   this->particle_track(),
                                   this->direction(),
                                   this->secondary_allocator());

    int                   num_secondaries = 0;
    std::map<double, int> energy_to_count;
    std::vector<double>   energy;
    std::vector<int>      count;

    // Loop over many particles
    for (int i = 0; i < num_samples; ++i)
    {
        Interaction out = interact(rng_engine);
        SCOPED_TRACE(out);
        ASSERT_TRUE(out);
        this->check_energy_conservation(out);
        num_secondaries += out.secondaries.size();

        for (const auto& secondary : out.secondaries)
        {
            // Increment the count of the discrete sampled energy
            energy_to_count[secondary.energy.value()]++;
        }
    }
    EXPECT_EQ(max_secondary * num_samples,
              this->secondary_allocator().get().size());
    EXPECT_EQ(10007, num_secondaries);

    for (const auto& it : energy_to_count)
    {
        energy.push_back(it.first);
        count.push_back(it.second);
    }
    const double expected_energy[] = {
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
    const int expected_count[]
        = {2, 1, 1, 1, 2, 2525, 2228, 4358, 337, 181, 361, 10};
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_EQ(expected_count, count);
}

TEST_F(LivermorePEInteractorTest, model)
{
    using celeritas::Collection;

    // Model is constructed with device pointers
    if (!celeritas::device())
    {
        SKIP("CUDA is disabled");
    }

    // Create physics tables
    ImportPhysicsTable xs_lo;
    xs_lo.table_type = ImportTableType::lambda;
    xs_lo.physics_vectors.push_back(
        {ImportPhysicsVectorType::log, {1e-2, 1, 1e2}, {1e-1, 1e-3, 1e-5}});

    ImportPhysicsTable xs_hi;
    xs_hi.table_type = ImportTableType::lambda_prim;
    xs_hi.physics_vectors.push_back(
        {ImportPhysicsVectorType::log, {1e2, 1e4, 1e6}, {1e-3, 1e-3, 1e-3}});

    // Add atomic relaxation data
    relax_inp_.is_auger_enabled = true;
    set_relaxation_params(relax_inp_);

    PhotoelectricProcess process(this->get_particle_params(),
                                 xs_lo,
                                 xs_hi,
                                 livermore_params_,
                                 relax_params_,
                                 10);

    Applicability range    = {MaterialId{0},
                           this->particle_params().find(pdg::gamma()),
                           celeritas::zero_quantity(),
                           celeritas::max_quantity()};
    auto          builders = process.step_limits(range);

    Collection<double, Ownership::value, MemSpace::host> real_storage;
    Collection<celeritas::XsGridData, Ownership::value, MemSpace::host>
        grid_storage;

    ValueGridInserter insert(&real_storage, &grid_storage);
    builders[int(celeritas::ValueGridType::macro_xs)]->build(insert);
    EXPECT_EQ(1, grid_storage.size());

    // Test cross sections calculated from tables
    Collection<double, Ownership::const_reference, MemSpace::host> real_ref{
        real_storage};
    celeritas::XsCalculator calc_xs(
        grid_storage[ValueGridInserter::XsIndex{0}], real_ref);
    EXPECT_SOFT_EQ(0.1, calc_xs(MevEnergy{1e-3}));
    EXPECT_SOFT_EQ(1e-5, calc_xs(MevEnergy{1e2}));
    EXPECT_SOFT_EQ(1e-9, calc_xs(MevEnergy{1e6}));

    // Construct the models associated with the photoelectric effect
    ModelIdGenerator next_id;
    auto models = process.build_models(next_id);
    EXPECT_EQ(1, models.size());

    auto livermore_pe = models.front();
    EXPECT_EQ(ModelId{0}, livermore_pe->model_id());

    // Get the particle types and energy ranges this model applies to
    auto set_applic = livermore_pe->applicability();
    EXPECT_EQ(1, set_applic.size());

    auto applic = *set_applic.begin();
    EXPECT_EQ(MaterialId{}, applic.material);
    EXPECT_EQ(ParticleId{1}, applic.particle);
    EXPECT_EQ(celeritas::zero_quantity(), applic.lower);
    EXPECT_EQ(celeritas::max_quantity(), applic.upper);
}

TEST_F(LivermorePEInteractorTest, macro_xs)
{
    using celeritas::units::MevEnergy;

    auto material = this->material_track().material_view();
    LivermorePEMacroXsCalculator calc_macro_xs(pointers_, material);

    int    num_vals = 20;
    double loge_min = std::log(1.e-4);
    double loge_max = std::log(1.e6);
    double delta    = (loge_max - loge_min) / (num_vals - 1);
    double loge     = loge_min;

    std::vector<double> energy;
    std::vector<double> macro_xs;

    // Loop over energies
    for (int i = 0; i < num_vals; ++i)
    {
        double e = std::exp(loge);
        energy.push_back(e);
        macro_xs.push_back(calc_macro_xs(MevEnergy{e}));
        loge += delta;
    }
    const double expected_macro_xs[]
        = {9.235615290944,     17.56658325086,     1.161217594282,
           0.4108511065363,    0.01515608909912,   0.0004000659204694,
           9.083754758322e-06, 2.449452106704e-07, 1.800625084911e-08,
           3.188458732396e-09, 8.028833591133e-10, 2.2700912115e-10,
           6.653075041804e-11, 1.971081007251e-11, 5.85857761177e-12,
           1.743005702864e-12, 5.187166124179e-13, 1.543827005416e-13,
           4.594922185898e-14, 1.367605938008e-14};
    EXPECT_VEC_SOFT_EQ(expected_macro_xs, macro_xs);
}

TEST_F(LivermorePEInteractorTest, max_secondaries)
{
    using celeritas::AtomicRelaxElement;
    using celeritas::AtomicRelaxSubshell;
    using celeritas::AtomicRelaxTransition;

    // For an element with n shells of transition data, the maximum number of
    // secondaries created can be upper-bounded as n if there are only
    // radiative transitions and 2^n - 1 if there are non-radiative transitions
    // for the hypothetical worst case where for a given vacancy the
    // transitions always originate from the next subshell up
    unsigned int num_shells        = 20;
    unsigned int upper_bound_fluor = num_shells;
    unsigned int upper_bound_auger = std::exp2(num_shells) - 1;

    AtomicRelaxElement                   el;
    std::vector<AtomicRelaxSubshell>     shell_storage(num_shells);
    celeritas::Span<AtomicRelaxSubshell> shells = make_span(shell_storage);
    {
        // One radiative transition per subshell, each one originating in the
        // next subshell up
        std::vector<AtomicRelaxTransition> transition_storage;
        transition_storage.reserve(num_shells);
        for (auto i : celeritas::range(num_shells))
        {
            transition_storage.push_back(
                {SubshellId{i + 1}, SubshellId{}, 1, 1});
            shells[i].transitions = {transition_storage.data() + i, 1};
        }
        el.shells   = shells;
        auto result = celeritas::detail::calc_max_secondaries(
            el, MevEnergy{0}, MevEnergy{0});
        EXPECT_EQ(upper_bound_fluor, result);
    }
    {
        // num_shells - subshell_id non-radiative transitions per subshell, one
        // originating in each of the higher subshells
        std::vector<AtomicRelaxTransition> transition_storage;
        transition_storage.reserve(num_shells * (num_shells + 1) / 2);
        for (auto i : celeritas::range(num_shells))
        {
            auto start = transition_storage.size();
            for (auto j : celeritas::range(i, num_shells))
            {
                transition_storage.push_back({SubshellId{j + 1},
                                              SubshellId{j + 1},
                                              1. / (num_shells - i),
                                              1});
            }
            shells[i].transitions
                = {transition_storage.data() + start,
                   transition_storage.data() + transition_storage.size()};
        }
        el.shells   = shells;
        auto result = celeritas::detail::calc_max_secondaries(
            el, MevEnergy{0}, MevEnergy{0});
        EXPECT_EQ(upper_bound_auger, result);
    }
    {
        relax_inp_.is_auger_enabled = true;
        relax_inp_.electron_cut     = MevEnergy{1.e-3};
        relax_inp_.gamma_cut        = MevEnergy{1.e-3};
        set_relaxation_params(relax_inp_);
        EXPECT_EQ(1, relax_params_->host_pointers().elements[0].max_secondary);

        relax_inp_.electron_cut = MevEnergy{1.e-4};
        relax_inp_.gamma_cut    = MevEnergy{1.e-4};
        set_relaxation_params(relax_inp_);
        EXPECT_EQ(3, relax_params_->host_pointers().elements[0].max_secondary);
    }
}
