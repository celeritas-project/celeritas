//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/CombinedBrem.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/CombinedBremInteractor.hh"
#include "celeritas/em/model/CombinedBremModel.hh"
#include "celeritas/io/SeltzerBergerReader.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/InteractionIO.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using Energy = units::MevEnergy;
using EnergySq = SBEnergyDistHelper::EnergySq;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CombinedBremTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using namespace constants;
        using namespace units;

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.elements
            = {{AtomicNumber{29}, units::AmuMass{63.546}, {}, "Cu"}};
        mat_inp.materials = {
            {native_value_from(MolCcDensity{0.141}),
             293.0,
             MatterState::solid,
             {{ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(mat_inp);

        // Set up Seltzer-Berger cross section data
        std::string data_path = this->test_data_path("celeritas", "");
        SeltzerBergerReader read_element_data(data_path.c_str());

        // Create mock import data
        {
            ImportProcess ip_electron = this->make_import_process(
                pdg::electron(),
                pdg::gamma(),
                ImportProcessClass::e_brems,
                {ImportModelClass::e_brems_sb, ImportModelClass::e_brems_lpm},
                {{0, 1e3}, {1e3, 1e12}});
            ImportProcess ip_positron = ip_electron;
            ip_positron.particle_pdg = pdg::positron().get();
            this->set_imported_processes(
                {std::move(ip_electron), std::move(ip_positron)});
        }

        // Construct SeltzerBergerModel and set host data
        model_ = std::make_shared<CombinedBremModel>(ActionId{0},
                                                     *this->particle_params(),
                                                     *this->material_params(),
                                                     this->imported_processes(),
                                                     read_element_data,
                                                     true);

        // Set cutoffs
        CutoffParams::Input input;
        CutoffParams::MaterialCutoffs material_cutoffs;
        material_cutoffs.push_back({MevEnergy{0.02064384}, 0.07});
        input.materials = this->material_params();
        input.particles = this->particle_params();
        input.cutoffs.insert({pdg::gamma(), material_cutoffs});
        this->set_cutoff_params(input);

        // Set default particle to incident 1 MeV photon in Cu
        this->set_inc_particle(pdg::electron(), MevEnergy{1.0});
        this->set_inc_direction({0, 0, 1});
        this->set_material("Cu");
    }

    EnergySq density_correction(PhysMatId matid, Energy e) const
    {
        CELER_EXPECT(matid);
        CELER_EXPECT(e > zero_quantity());
        using namespace constants;

        auto mat = this->material_params()->get(matid);
        constexpr auto migdal = 4 * pi * r_electron
                                * ipow<2>(lambdabar_electron);

        real_type density_factor = mat.electron_density() * migdal;
        return EnergySq{density_factor * ipow<2>(e.value())};
    }

    void sanity_check(Interaction const& interaction) const
    {
        EXPECT_EQ(Action::scattered, interaction.action);
    }

  protected:
    std::shared_ptr<CombinedBremModel> model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CombinedBremTest, basic_seltzer_berger)
{
    // Reserve 4 secondaries, one for each sample
    int const num_samples = 4;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().material_record();
    auto cutoffs = this->cutoff_params()->get(PhysMatId{0});

    // Create the interactor
    CombinedBremInteractor interact(model_->host_ref(),
                                    this->particle_track(),
                                    this->direction(),
                                    cutoffs,
                                    this->secondary_allocator(),
                                    material_view,
                                    ElementComponentId{0});
    RandomEngine& rng_engine = this->rng();

    // Produce two samples from the original/incident photon
    std::vector<double> angle;
    std::vector<double> energy;

    // Loop number of samples
    for (int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        energy.push_back(result.secondaries[0].energy.value());
        angle.push_back(dot_product(result.direction,
                                    result.secondaries.front().direction));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    double const expected_angle[] = {0.959441513277674,
                                     0.994350429950924,
                                     0.968866136008621,
                                     0.961582855967571};

    double const expected_energy[] = {0.0349225070114679,
                                      0.0316182310804369,
                                      0.0838794010486177,
                                      0.106195186929141};
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(CombinedBremTest, basic_relativistic_brem)
{
    int const num_samples = 4;

    // Reserve  num_samples secondaries;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().material_record();
    auto cutoffs = this->cutoff_params()->get(PhysMatId{0});

    // Set the incident particle energy
    this->set_inc_particle(pdg::electron(), MevEnergy{25000});

    // Create the interactor
    CombinedBremInteractor interact(model_->host_ref(),
                                    this->particle_track(),
                                    this->direction(),
                                    cutoffs,
                                    this->secondary_allocator(),
                                    material_view,
                                    ElementComponentId{0});

    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original incident angle/energy
    std::vector<double> angle;
    std::vector<double> energy;

    for (int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        energy.push_back(result.secondaries[0].energy.value());
        angle.push_back(dot_product(result.direction,
                                    result.secondaries.back().direction));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.

    double const expected_energy[] = {
        18844.5999305425, 42.185863858534, 3991.9107959354, 212.273682952066};

    double const expected_angle[] = {0.999999972054405,
                                     0.999999999587026,
                                     0.999999999684891,
                                     0.999999999474844};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(CombinedBremTest, stress_test_combined)
{
    int const num_samples = 10000;
    std::vector<double> avg_engine_samples;
    std::vector<double> avg_energy_samples;

    // Views
    auto cutoffs = this->cutoff_params()->get(PhysMatId{0});
    auto material_view = this->material_track().material_record();

    // Loop over a set of incident gamma energies
    real_type const test_energy[]
        = {1.5, 5, 10, 50, 100, 1000, 1e+4, 1e+5, 1e+6};

    for (auto particle : {pdg::electron(), pdg::positron()})
    {
        for (real_type inc_e : test_energy)
        {
            SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));

            RandomEngine& rng_engine = this->rng();
            RandomEngine::size_type num_particles_sampled = 0;
            double tot_energy_sampled = 0;

            // Loop over several incident directions
            for (Real3 const& inc_dir : {Real3{0, 0, 1},
                                         Real3{1, 0, 0},
                                         Real3{1e-9, 0, 1},
                                         Real3{1, 1, 1}})
            {
                this->set_inc_direction(inc_dir);
                this->resize_secondaries(num_samples);

                // Create interactor
                this->set_inc_particle(particle, MevEnergy{inc_e});
                CombinedBremInteractor interact(model_->host_ref(),
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
                    tot_energy_sampled += result.secondaries[0].energy.value();
                }
                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(double(rng_engine.count())
                                         / double(num_particles_sampled));
            avg_energy_samples.push_back(tot_energy_sampled
                                         / double(num_particles_sampled));
        }
    }

    // Gold values for average number of calls to RNG
    static double const expected_avg_engine_samples[] = {14.088,
                                                         13.2402,
                                                         12.9641,
                                                         12.5832,
                                                         12.4988,
                                                         12.31,
                                                         12.4381,
                                                         13.2552,
                                                         15.3604,
                                                         14.2257,
                                                         13.2616,
                                                         12.9286,
                                                         12.5763,
                                                         12.5076,
                                                         12.3059,
                                                         12.4207,
                                                         13.2906,
                                                         15.3809};
    static double const expected_avg_energy_samples[] = {0.20338654094171,
                                                         0.53173619503507,
                                                         0.99638562846318,
                                                         4.4359411867158,
                                                         8.7590072534526,
                                                         85.769352932541,
                                                         905.74010590236,
                                                         10724.127172904,
                                                         149587.47778726,
                                                         0.18927693789863,
                                                         0.52259561993542,
                                                         0.98783539434815,
                                                         4.4286338859014,
                                                         8.495313663667,
                                                         85.853825010643,
                                                         917.26619467326,
                                                         10760.512214274,
                                                         146925.51055711};

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
    EXPECT_VEC_SOFT_EQ(expected_avg_energy_samples, avg_energy_samples);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
