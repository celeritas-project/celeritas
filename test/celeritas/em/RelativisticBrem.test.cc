//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/RelativisticBrem.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/interactor/RelativisticBremInteractor.hh"
#include "celeritas/em/model/RelativisticBremModel.hh"
#include "celeritas/em/xs/RBDiffXsCalculator.hh"
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
// TEST HARNESS
//---------------------------------------------------------------------------//

class RelativisticBremTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using namespace units;

        // Set up shared material data
        MaterialParams::Input mi;
        mi.elements = {{AtomicNumber{82}, AmuMass{207.2}, {}, "Pb"}};
        mi.materials = {{native_value_from(MolCcDensity{0.05477}),
                         293.15,
                         MatterState::solid,
                         {{ElementId{0}, 1.0}},
                         "Pb"}};

        // Set default material to potassium
        this->set_material_params(mi);

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

        auto const& particles = *this->particle_params();

        // Construct RelativisticBremModel
        model_ = std::make_shared<RelativisticBremModel>(
            ActionId{0},
            particles,
            *this->material_params(),
            this->imported_processes(),
            false);

        // Construct RelativisticBremModel with LPM
        model_lpm_ = std::make_shared<RelativisticBremModel>(
            ActionId{0},
            particles,
            *this->material_params(),
            this->imported_processes(),
            true);

        // Set cutoffs: photon energy thresholds and range cut for Pb
        CutoffParams::Input input;
        CutoffParams::MaterialCutoffs material_cutoffs;
        material_cutoffs.push_back({MevEnergy{0.0945861}, 0.07});
        input.materials = this->material_params();
        input.particles = this->particle_params();
        input.cutoffs.insert({pdg::gamma(), material_cutoffs});
        this->set_cutoff_params(input);

        // Set default particle to incident 25 GeV electron
        this->set_inc_particle(pdg::positron(), MevEnergy{25000});
        this->set_inc_direction({0, 0, 1});
        this->set_material("Pb");
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Check secondaries (bremsstrahlung photon)
        ASSERT_EQ(1, interaction.secondaries.size());

        auto const& gamma = interaction.secondaries.front();
        EXPECT_TRUE(gamma);
        EXPECT_EQ(model_->host_ref().ids.gamma, gamma.particle_id);

        // Check conservation
        this->check_conservation(interaction);
    }

  protected:
    std::shared_ptr<RelativisticBremModel> model_;
    std::shared_ptr<RelativisticBremModel> model_lpm_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(RelativisticBremTest, dxsec)
{
    real_type const all_energy[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};

    // Production cuts
    auto material_view = this->material_track().material_record();

    // Create the differential cross section
    RBDiffXsCalculator dxsec_lpm(model_lpm_->host_ref(),
                                 this->particle_track(),
                                 material_view,
                                 ElementComponentId{0});

    // Create the differential cross section
    RBDiffXsCalculator dxsec(model_->host_ref(),
                             this->particle_track(),
                             material_view,
                             ElementComponentId{0});

    // Calculate cross section values at ten photon energy (MevEnergy) points
    std::vector<real_type> dxsec_value_lpm;
    std::vector<real_type> dxsec_value;

    for (real_type energy : all_energy)
    {
        real_type result = dxsec_lpm(MevEnergy{energy});
        dxsec_value_lpm.push_back(result);

        result = dxsec(MevEnergy{energy});
        dxsec_value.push_back(result);
    }

    // Note: these are "gold" differential cross sections by the photon energy.
    real_type const expected_dxsec_lpm[] = {3.15917865133079,
                                            2.24073793752395,
                                            1.7690465485807,
                                            1.9393060296929,
                                            2.31528198148389,
                                            2.89563604143589,
                                            3.27222667999604,
                                            3.5085408186385,
                                            3.48017265373814,
                                            3.41228707554124};

    real_type const expected_dxsec[] = {3.55000253342095,
                                        3.54986051043622,
                                        3.54943449138746,
                                        3.54872462599092,
                                        3.54730551901548,
                                        3.54305318862896,
                                        3.53598260644102,
                                        3.52190382370326,
                                        3.48016652710843,
                                        3.41226786120072};

    EXPECT_VEC_SOFT_EQ(expected_dxsec_lpm, dxsec_value_lpm);
    EXPECT_VEC_SOFT_EQ(expected_dxsec, dxsec_value);
}

TEST_F(RelativisticBremTest, basic_without_lpm)
{
    int const num_samples = 4;

    // Reserve  num_samples secondaries;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().material_record();
    auto cutoffs = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    RelativisticBremInteractor interact(model_->host_ref(),
                                        this->particle_track(),
                                        this->direction(),
                                        cutoffs,
                                        this->secondary_allocator(),
                                        material_view,
                                        ElementComponentId{0});

    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original incident angle/energy
    std::vector<real_type> angle;
    std::vector<real_type> energy;

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
    EXPECT_REAL_EQ(real_type(rng_engine.count()) / num_samples, 12);

    // Note: these are "gold" values based on the host RNG.

    real_type const expected_energy[] = {
        9.7121539090503, 16.1109589071687, 7.48863745059463, 8338.70226190511};

    real_type const expected_angle[] = {0.999999999858782,
                                        0.999999999999921,
                                        0.999999976998416,
                                        0.999999998137601};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(RelativisticBremTest, basic_with_lpm)
{
    int const num_samples = 4;

    // Reserve  num_samples secondaries;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().material_record();
    auto cutoffs = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    RelativisticBremInteractor interact(model_lpm_->host_ref(),
                                        this->particle_track(),
                                        this->direction(),
                                        cutoffs,
                                        this->secondary_allocator(),
                                        material_view,
                                        ElementComponentId{0});

    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original incident angle/energy
    std::vector<real_type> angle;
    std::vector<real_type> energy;

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

    real_type const expected_energy[] = {
        18872.4157243063, 43.6117832245235, 4030.31152398788, 217.621447606391};

    real_type const expected_angle[] = {0.999999971800136,
                                        0.999999999587026,
                                        0.999999999683752,
                                        0.999999999474844};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);
}

TEST_F(RelativisticBremTest, stress_with_lpm)
{
    int const num_samples = 1000;

    // Reserve  num_samples secondaries;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().material_record();
    auto cutoffs = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    RelativisticBremInteractor interact(model_lpm_->host_ref(),
                                        this->particle_track(),
                                        this->direction(),
                                        cutoffs,
                                        this->secondary_allocator(),
                                        material_view,
                                        ElementComponentId{0});

    RandomEngine& rng_engine = this->rng();

    // Produce samples from the original incident angle/energy
    real_type average_energy{0};
    Real3 average_angle{0, 0, 0};

    for (int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);

        average_energy += result.secondaries[0].energy.value();
        average_angle[0] += result.secondaries[0].direction[0];
        average_angle[1] += result.secondaries[0].direction[1];
        average_angle[2] += result.secondaries[0].direction[2];

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);
    }
    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    EXPECT_SOFT_EQ(average_energy / num_samples, 2932.1072998587733);
    EXPECT_SOFT_NEAR(
        average_angle[0] / num_samples, 3.3286986548662216e-06, 1e-8);
    EXPECT_SOFT_NEAR(
        average_angle[1] / num_samples, 1.3067055198983571e-06, 1e-8);
    EXPECT_SOFT_EQ(average_angle[2] / num_samples, 0.99999999899845182);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
