//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/Rayleigh.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/interactor/RayleighInteractor.hh"
#include "celeritas/em/model/RayleighModel.hh"
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

class RayleighInteractorTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using namespace constants;
        using namespace units;
        constexpr auto zero = zero_quantity();

        // Set up shared particle data for RayleighModel
        Base::set_particle_params(
            {{"gamma", pdg::gamma(), zero, zero, stable_decay_constant}});
        auto const& particles = *this->particle_params();
        model_ref_.gamma = particles.find(pdg::gamma());

        // Setup MaterialView
        MaterialParams::Input inp;
        inp.elements = {{AtomicNumber{8}, units::AmuMass{15.999}, {}, "O"},
                        {AtomicNumber{74}, units::AmuMass{183.84}, {}, "W"},
                        {AtomicNumber{82}, units::AmuMass{207.2}, {}, "Pb"}};
        inp.materials = {
            {native_value_from(MolCcDensity{1.0}),
             293.0,
             MatterState::solid,
             {{ElementId{0}, 0.5}, {ElementId{1}, 0.3}, {ElementId{2}, 0.2}},
             "PbWO"},
        };
        this->set_material_params(inp);

        // Imported process data needed to construct the model (with empty
        // physics tables, which are not needed for the interactor)
        this->set_imported_processes(
            {this->make_import_process(pdg::gamma(),
                                       {},
                                       ImportProcessClass::rayleigh,
                                       {ImportModelClass::livermore_rayleigh},
                                       {{0, 1e12}})});

        // Construct RayleighModel and save the host data reference
        model_ = std::make_shared<RayleighModel>(ActionId{0},
                                                 particles,
                                                 *this->material_params(),
                                                 this->imported_processes());
        model_ref_ = model_->host_ref();

        // Set default particle to incident 1 MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{1.0});
        this->set_inc_direction({0, 0, 1});
        this->set_material("PbWO");
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Check change to parent track - coherent scattering
        EXPECT_EQ(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_EQ(Action::scattered, interaction.action);
    }

  protected:
    std::shared_ptr<RayleighModel> model_;
    RayleighRef model_ref_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RayleighInteractorTest, basic)
{
    // Sample an element (TODO: add ElementSelector)
    ElementId el_id{0};

    std::vector<real_type> angle;
    std::vector<unsigned long int> rng_counts;

    // Sample scattering angle and count rng used for each incident energy
    for (real_type inc_e : {1e-5, 1e-4, 0.001, 0.01, 0.1, 1., 10., 100., 1000.})
    {
        RandomEngine& rng_engine = this->rng();

        // Set the incident particle energy
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

        // Create the interactor
        RayleighInteractor interact(this->model_->host_ref(),
                                    this->particle_track(),
                                    this->direction(),
                                    el_id);

        // Produce a sample from the original/incident photon
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        angle.push_back(dot_product(result.direction, this->direction()));
        rng_counts.push_back(rng_engine.count());
    }

    real_type const expected_angle[] = {0.383668498876068,
                                        -0.99294588967104,
                                        0.780467077338104,
                                        0.985521422599946,
                                        0.875273769840553,
                                        0.999674148324654,
                                        0.999998842967848,
                                        0.99999999296325,
                                        0.999999999919784};

    unsigned long int const expected_rng_counts[]
        = {14, 8, 8, 8, 8, 8, 8, 8, 8};

    EXPECT_VEC_SOFT_EQ(expected_angle, angle);
    EXPECT_VEC_EQ(expected_rng_counts, rng_counts);
}

TEST_F(RayleighInteractorTest, stress_test)
{
    int const num_samples = 8192;

    // Sample an element
    ElementId el_id{0};

    std::vector<real_type> average_angle;
    std::vector<real_type> average_rng_counts;

    // Sample scattering angle and count rng used for each incident energy
    for (real_type inc_e : {1e-5, 1e-4, 0.001, 0.01, 0.1, 1., 10., 100., 1000.})
    {
        // Set the incident particle energy
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

        // Reset the rng counter
        RandomEngine& rng_engine = this->rng();

        // Create the interactor
        RayleighInteractor interact(this->model_->host_ref(),
                                    this->particle_track(),
                                    this->direction(),
                                    el_id);

        // Produce num_samples from the original/incident photon
        real_type sum_angle = 0;
        for ([[maybe_unused]] auto i : range(num_samples))
        {
            Interaction result = interact(rng_engine);
            SCOPED_TRACE(result);
            this->sanity_check(result);
            rng_engine.count();
            sum_angle += dot_product(result.direction, this->direction());
        }

        average_rng_counts.push_back(real_type(rng_engine.count())
                                     / real_type(num_samples));
        average_angle.push_back(sum_angle / num_samples);
    }

    real_type const expected_average_rng_counts[] = {10.943603515625,
                                                     11.01025390625,
                                                     11.08935546875,
                                                     9.82080078125,
                                                     8.308349609375,
                                                     8.002197265625,
                                                     8,
                                                     8,
                                                     8};

    real_type const expected_average_angle[] = {0.00231121922009911,
                                                0.00899744556924152,
                                                0.00779010297910534,
                                                0.583035907797808,
                                                0.951988493573674,
                                                0.999415919902184,
                                                0.999994055745254,
                                                0.999999938196652,
                                                0.999999999411519};

    EXPECT_VEC_SOFT_EQ(expected_average_rng_counts, average_rng_counts);
    EXPECT_VEC_SOFT_EQ(expected_average_angle, average_angle);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
