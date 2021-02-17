//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePE.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/detail/LivermorePEInteractor.hh"

#include <fstream>
#include <map>
#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "comm/Device.hh"
#include "io/AtomicRelaxationReader.hh"
#include "io/LivermorePEParamsReader.hh"
#include "physics/base/Units.hh"
#include "physics/em/AtomicRelaxationParams.hh"
#include "physics/em/LivermorePEModel.hh"
#include "physics/em/LivermorePEParams.hh"
#include "physics/em/PhotoelectricProcess.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::AtomicRelaxationParams;
using celeritas::AtomicRelaxationReader;
using celeritas::ElementId;
using celeritas::LivermorePEParams;
using celeritas::LivermorePEParamsReader;
using celeritas::PhotoelectricProcess;
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

  protected:
    AtomicRelaxationParams::Input           relax_inp_;
    std::shared_ptr<AtomicRelaxationParams> relax_params_;
    std::shared_ptr<LivermorePEParams>      livermore_params_;
    celeritas::detail::LivermorePEPointers  pointers_;
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
                                           el_id,
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
    this->resize_secondaries(16 * num_samples);

    // Sampled element
    ElementId el_id{0};

    // Add atomic relaxation data
    relax_inp_.is_auger_enabled = true;
    set_relaxation_params(relax_inp_);
    pointers_.atomic_relaxation = relax_params_->host_pointers();

    // Create the interactor
    LivermorePEInteractor interact(pointers_,
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
    EXPECT_EQ(16 * num_samples, this->secondary_allocator().get().size());
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
    this->resize_secondaries(5 * num_samples);

    // Sampled element
    ElementId el_id{0};

    // Add atomic relaxation data
    relax_inp_.is_auger_enabled = false;
    set_relaxation_params(relax_inp_);
    pointers_.atomic_relaxation = relax_params_->host_pointers();

    // Create the interactor
    LivermorePEInteractor interact(pointers_,
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
    EXPECT_EQ(5 * num_samples, this->secondary_allocator().get().size());
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
    // Model is constructed with device pointers
    if (!celeritas::device())
    {
        SKIP("CUDA is disabled");
    }

    PhotoelectricProcess process(
        this->get_particle_params(), livermore_params_, relax_params_);
    ModelIdGenerator     next_id;

    // Construct the models associated with the photoelectric effect
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
