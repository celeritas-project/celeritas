//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/CoulombScattering.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/interactor/CoulombScatteringInteractor.hh"
#include "celeritas/em/model/CoulombScatteringModel.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/em/process/CoulombScatteringProcess.hh"
#include "celeritas/em/xs/WentzelTransportXsCalculator.hh"
#include "celeritas/io/ImportParameters.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/InteractionIO.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class CoulombScatteringTest : public InteractorHostTestBase
{
  protected:
    void SetUp() override
    {
        using namespace celeritas::units;
        using constants::stable_decay_constant;

        // Need to include protons
        constexpr units::MevMass emass{0.5109989461};
        ParticleParams::Input par_inp = {
            {"electron",
             pdg::electron(),
             emass,
             ElementaryCharge{-1},
             stable_decay_constant},
            {"positron",
             pdg::positron(),
             emass,
             ElementaryCharge{1},
             stable_decay_constant},
            {"proton",
             pdg::proton(),
             units::MevMass{938.28},
             ElementaryCharge{1},
             stable_decay_constant},
        };
        this->set_particle_params(std::move(par_inp));

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.isotopes
            = {{AtomicNumber{29}, AtomicNumber{63}, MevMass{58618.5}, "63Cu"},
               {AtomicNumber{29}, AtomicNumber{65}, MevMass{60479.8}, "65Cu"}};
        mat_inp.elements = {{AtomicNumber{29},
                             AmuMass{63.546},
                             {{IsotopeId{0}, 0.692}, {IsotopeId{1}, 0.308}},
                             "Cu"}};
        mat_inp.materials = {
            {native_value_from(MolCcDensity{0.141}),
             293.0,
             MatterState::solid,
             {{ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(mat_inp);

        // Create mock import data
        {
            ImportProcess ip_electron = this->make_import_process(
                pdg::electron(),
                {},
                ImportProcessClass::coulomb_scat,
                {ImportModelClass::e_coulomb_scattering});
            ImportProcess ip_positron = ip_electron;
            ip_positron.particle_pdg = pdg::positron().get();
            this->set_imported_processes(
                {std::move(ip_electron), std::move(ip_positron)});
        }

        // Default to single scattering
        WentzelOKVIParams::Options options;
        options.is_combined = false;
        options.polar_angle_limit = 0;
        wentzel_ = std::make_shared<WentzelOKVIParams>(this->material_params(),
                                                       options);

        model_ = std::make_shared<CoulombScatteringModel>(
            ActionId{0}, *this->particle_params(), this->imported_processes());

        // Set cutoffs
        CutoffParams::Input input;
        CutoffParams::MaterialCutoffs material_cutoffs;
        // TODO: Use realistic cutoff / material with high cutoff
        material_cutoffs.push_back({MevEnergy{0.5}, 0.07});
        input.materials = this->material_params();
        input.particles = this->particle_params();
        input.cutoffs.insert({pdg::electron(), material_cutoffs});
        input.cutoffs.insert({pdg::positron(), material_cutoffs});
        input.cutoffs.insert({pdg::proton(), material_cutoffs});
        this->set_cutoff_params(input);

        // Set incident particle to be an electron at 200 MeV
        this->set_inc_particle(pdg::electron(), MevEnergy{200.0});
        this->set_inc_direction({0, 0, 1});
        this->set_material("Cu");
    }

    void sanity_check(Interaction const& interaction) const
    {
        SCOPED_TRACE(interaction);

        // Check change to parent track
        EXPECT_GE(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(interaction.direction));
        EXPECT_EQ(Action::scattered, interaction.action);

        // Check secondaries
        EXPECT_TRUE(interaction.secondaries.empty());

        // Non-zero energy deposit in material so momentum isn't conserved
        this->check_energy_conservation(interaction);
    }

  protected:
    std::shared_ptr<WentzelOKVIParams> wentzel_;
    std::shared_ptr<CoulombScatteringModel> model_;
    IsotopeComponentId isocomp_id_{0};
    ElementComponentId elcomp_id_{0};
    ElementId el_id_{0};
    MaterialId mat_id_{0};
};

TEST_F(CoulombScatteringTest, helper)
{
    struct Result
    {
        using VecReal = std::vector<real_type>;

        VecReal screen_z;
        VecReal scaled_kin_factor;
        VecReal costheta_max_elec;
        VecReal costheta_max_nuc;
        VecReal xs_elec;
        VecReal xs_nuc;
        VecReal xs_ratio;
    };

    auto const material = this->material_track().make_material_view();
    AtomicNumber const target_z
        = this->material_params()->get(el_id_).atomic_number();

    MevEnergy const cutoff = this->cutoff_params()->get(mat_id_).energy(
        this->particle_track().particle_id());

    real_type const costheta_max = model_->host_ref().costheta_max();

    Result result;
    for (real_type energy : {50, 100, 200, 1000, 13000})
    {
        this->set_inc_particle(pdg::electron(), MevEnergy{energy});

        WentzelHelper helper(this->particle_track(),
                             material,
                             target_z,
                             wentzel_->host_ref().params,
                             model_->host_ref().ids,
                             cutoff);

        result.screen_z.push_back(helper.screening_coefficient());
        // Scale the xs factor by 1 / r_e^2 so the values will be large enough
        // for the soft equivalence comparison to catch any differences
        result.scaled_kin_factor.push_back(helper.kin_factor()
                                           / ipow<2>(constants::r_electron));
        result.costheta_max_elec.push_back(helper.costheta_max_electron());
        real_type const costheta_max_nuc = helper.costheta_max_nuclear();
        result.costheta_max_nuc.push_back(costheta_max_nuc);
        result.xs_elec.push_back(
            helper.calc_xs_electron(costheta_max_nuc, costheta_max)
            / units::barn);
        result.xs_nuc.push_back(
            helper.calc_xs_nuclear(costheta_max_nuc, costheta_max)
            / units::barn);
        result.xs_ratio.push_back(
            helper.calc_xs_ratio(costheta_max_nuc, costheta_max));
    }

    static double const expected_screen_z[] = {2.1181757502465e-08,
                                               5.3641196710457e-09,
                                               1.3498490873627e-09,
                                               5.4280909096648e-11,
                                               3.2158426877075e-13};
    static double const expected_scaled_kin_factor[] = {0.018652406309778,
                                                        0.0047099159161888,
                                                        0.0011834423911797,
                                                        4.7530717872407e-05,
                                                        2.8151208086621e-07};
    static double const expected_costheta_max_elec[] = {0.99989885103277,
                                                        0.99997458240728,
                                                        0.99999362912075,
                                                        0.99999974463379,
                                                        0.99999999848823};
    static double const expected_costheta_max_nuc[] = {1, 1, 1, 1, 1};
    static double const expected_xs_elec[] = {40826.46816866,
                                              40708.229862005,
                                              40647.018860182,
                                              40596.955206725,
                                              40585.257368735};
    static double const expected_xs_nuc[] = {1184463.4246675,
                                             1181036.9405781,
                                             1179263.0534548,
                                             1177812.2021574,
                                             1177473.1955599};
    static double const expected_xs_ratio[] = {0.033319844069031,
                                               0.033319738720425,
                                               0.033319684608429,
                                               0.033319640583261,
                                               0.03331963032739};
    EXPECT_VEC_SOFT_EQ(expected_screen_z, result.screen_z);
    EXPECT_VEC_SOFT_EQ(expected_scaled_kin_factor, result.scaled_kin_factor);
    EXPECT_VEC_SOFT_EQ(expected_costheta_max_elec, result.costheta_max_elec);
    EXPECT_VEC_SOFT_EQ(expected_costheta_max_nuc, result.costheta_max_nuc);
    EXPECT_VEC_SOFT_EQ(expected_xs_elec, result.xs_elec);
    EXPECT_VEC_SOFT_EQ(expected_xs_nuc, result.xs_nuc);
    EXPECT_VEC_SOFT_EQ(expected_xs_ratio, result.xs_ratio);
}

TEST_F(CoulombScatteringTest, mott_xs)
{
    MottElementData const& element_data
        = wentzel_->host_ref().elem_data[el_id_];
    MottRatioCalculator xsec(element_data,
                             sqrt(this->particle_track().beta_sq()));

    static real_type const cos_ts[]
        = {1, 0.9, 0.5, 0.21, 0, -0.1, -0.6, -0.7, -0.9, -1};
    static real_type const expected_xsecs[] = {0.99997507022045,
                                               1.090740570075,
                                               0.98638178782896,
                                               0.83702240402998,
                                               0.71099171311683,
                                               0.64712379625713,
                                               0.30071752615308,
                                               0.22722448378001,
                                               0.07702815350459,
                                               0.00051427465924958};

    std::vector<real_type> xsecs;
    for (real_type cos_t : cos_ts)
    {
        xsecs.push_back(xsec(cos_t));
    }

    EXPECT_VEC_SOFT_EQ(xsecs, expected_xsecs);
}

TEST_F(CoulombScatteringTest, wokvi_transport_xs)
{
    auto const material = this->material_track().make_material_view();
    AtomicNumber const z = this->material_params()->get(el_id_).atomic_number();

    // Incident particle energy cutoff
    MevEnergy const cutoff = this->cutoff_params()->get(mat_id_).energy(
        this->particle_track().particle_id());

    std::vector<real_type> xs;
    for (real_type energy : {100, 200, 1000, 100000, 1000000})
    {
        this->set_inc_particle(pdg::electron(), MevEnergy{energy});
        auto const& particle = this->particle_track();

        WentzelHelper helper(particle,
                             material,
                             z,
                             wentzel_->host_ref().params,
                             model_->host_ref().ids,
                             cutoff);
        WentzelTransportXsCalculator calc_transport_xs(particle, helper);

        for (real_type costheta_max : {-1.0, -0.5, 0.0, 0.5, 0.75, 0.99, 1.0})
        {
            // Get cross section in barns
            xs.push_back(calc_transport_xs(costheta_max) / units::barn);
        }
    }
    static double const expected_xs[] = {0.18738907324438,
                                         0.18698029857321,
                                         0.18529403401504,
                                         0.1804875329214,
                                         0.17432530107014,
                                         0.14071448472406,
                                         0,
                                         0.050844259956663,
                                         0.050741561907078,
                                         0.050317873900199,
                                         0.049110176080819,
                                         0.047561822391017,
                                         0.039116564509577,
                                         0,
                                         0.00239379259103,
                                         0.0023896680893247,
                                         0.0023726516350529,
                                         0.0023241469141481,
                                         0.0022619603107973,
                                         0.0019227725825426,
                                         0,
                                         3.4052045960474e-07,
                                         3.4010759295258e-07,
                                         3.3840422701414e-07,
                                         3.3354884935259e-07,
                                         3.2732389915557e-07,
                                         2.9337081738993e-07,
                                         0,
                                         3.9098094802963e-09,
                                         3.9056807758002e-09,
                                         3.8886469597418e-09,
                                         3.8400927365321e-09,
                                         3.7778426619946e-09,
                                         3.4383087213522e-09,
                                         0};
    EXPECT_VEC_SOFT_EQ(expected_xs, xs);
}

TEST_F(CoulombScatteringTest, simple_scattering)
{
    int const num_samples = 4;

    auto const material = this->material_track().make_material_view();
    IsotopeView const isotope
        = material.make_element_view(elcomp_id_).make_isotope_view(isocomp_id_);
    auto cutoffs = this->cutoff_params()->get(mat_id_);

    RandomEngine& rng_engine = this->rng();

    std::vector<real_type> cos_theta;
    std::vector<real_type> delta_energy;

    std::vector<real_type> energies{0.2, 1, 10, 100, 1000, 100000};
    for (auto energy : energies)
    {
        this->set_inc_particle(pdg::electron(), MevEnergy{energy});
        CoulombScatteringInteractor interact(model_->host_ref(),
                                             wentzel_->host_ref(),
                                             this->particle_track(),
                                             this->direction(),
                                             material,
                                             isotope,
                                             el_id_,
                                             cutoffs);

        for ([[maybe_unused]] int i : range(num_samples))
        {
            Interaction result = interact(rng_engine);
            SCOPED_TRACE(result);
            this->sanity_check(result);

            cos_theta.push_back(
                dot_product(this->direction(), result.direction));
            delta_energy.push_back(energy - result.energy.value());
        }
    }
    static double const expected_cos_theta[] = {1,
                                                0.9996601312603,
                                                0.99998628518524,
                                                0.9998960794451,
                                                1,
                                                0.99991152273528,
                                                0.99998247385882,
                                                0.99988427878015,
                                                1,
                                                0.99999970191006,
                                                0.99999637934855,
                                                0.99999885954183,
                                                0.999999997443,
                                                0.99999999406847,
                                                0.99999999849197,
                                                1,
                                                0.99999999992169,
                                                1,
                                                0.99999999209019,
                                                0.99999999999489,
                                                1,
                                                0.99999999999999,
                                                0.99999999999999,
                                                1};
    static double const expected_delta_energy[] = {0,
                                                   1.4170232209842e-09,
                                                   5.7181509527382e-11,
                                                   4.3327857968123e-10,
                                                   0,
                                                   3.0519519134131e-09,
                                                   6.0455007666604e-10,
                                                   3.9917101846143e-09,
                                                   0,
                                                   5.6049742624964e-10,
                                                   6.8078875870015e-09,
                                                   2.1443966602419e-09,
                                                   4.406643938637e-10,
                                                   1.0222294122286e-09,
                                                   2.5988811103161e-10,
                                                   0,
                                                   1.3371845852816e-09,
                                                   0,
                                                   1.350750835627e-07,
                                                   8.7311491370201e-11,
                                                   4.8021320253611e-10,
                                                   8.7311491370201e-10,
                                                   1.6734702512622e-09,
                                                   0};
    EXPECT_VEC_SOFT_EQ(expected_cos_theta, cos_theta);
    EXPECT_VEC_SOFT_EQ(expected_delta_energy, delta_energy);
}

TEST_F(CoulombScatteringTest, distribution)
{
    auto const material = this->material_track().make_material_view();
    IsotopeView const isotope
        = material.make_element_view(elcomp_id_).make_isotope_view(isocomp_id_);

    // TODO: Use proton ParticleId{2}
    MevEnergy const cutoff
        = this->cutoff_params()->get(mat_id_).energy(ParticleId{0});

    std::vector<real_type> avg_angles;

    for (real_type energy : {1, 50, 100, 200, 1000, 13000})
    {
        this->set_inc_particle(pdg::electron(), MevEnergy{energy});

        WentzelHelper helper(this->particle_track(),
                             material,
                             isotope.atomic_number(),
                             wentzel_->host_ref().params,
                             model_->host_ref().ids,
                             cutoff);
        WentzelDistribution sample_angle(wentzel_->host_ref(),
                                         helper,
                                         this->particle_track(),
                                         isotope,
                                         el_id_,
                                         helper.costheta_max_nuclear(),
                                         model_->host_ref().costheta_max());

        RandomEngine& rng_engine = this->rng();

        real_type avg_angle = 0;

        int const num_samples = 4096;
        for ([[maybe_unused]] int i : range(num_samples))
        {
            avg_angle += sample_angle(rng_engine);
        }

        avg_angle /= num_samples;
        avg_angles.push_back(avg_angle);
    }

    static double const expected_avg_angles[] = {0.99933909299229,
                                                 0.99999960697043,
                                                 0.99999986035881,
                                                 0.99999998052954,
                                                 0.99999999917037,
                                                 0.9999999999969};
    EXPECT_VEC_SOFT_EQ(expected_avg_angles, avg_angles);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
