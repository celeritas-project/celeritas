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
    using SPWentzel = std::shared_ptr<WentzelOKVIParams>;

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
        mat_inp.isotopes = {{AtomicNumber{29},
                             AtomicNumber{63},
                             MevEnergy{551.384},
                             MevEnergy{6.122},
                             MevEnergy{10.864},
                             MevMass{58618.5},
                             "63Cu"},
                            {AtomicNumber{29},
                             AtomicNumber{65},
                             MevEnergy{569.211},
                             MevEnergy{7.454},
                             MevEnergy{9.911},
                             MevMass{60479.8},
                             "65Cu"}};
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

        model_ = std::make_shared<CoulombScatteringModel>(
            ActionId{0},
            *this->particle_params(),
            *this->material_params(),
            this->imported_processes());

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

    SPWentzel make_wentzel_params(NuclearFormFactorType ff
                                  = NuclearFormFactorType::exponential)
    {
        // Default to single scattering
        WentzelOKVIParams::Options options;
        options.is_combined = false;
        options.polar_angle_limit = 0;
        options.form_factor = ff;
        return std::make_shared<WentzelOKVIParams>(this->material_params(),
                                                   options);
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
        VecReal cos_thetamax_elec;
        VecReal cos_thetamax_nuc;
        VecReal xs_elec;
        VecReal xs_nuc;
    };

    auto wentzel = this->make_wentzel_params();

    auto const material = this->material_track().make_material_view();
    AtomicNumber const target_z
        = this->material_params()->get(el_id_).atomic_number();

    MevEnergy const cutoff = this->cutoff_params()->get(mat_id_).energy(
        this->particle_track().particle_id());

    real_type const cos_thetamax = model_->host_ref().cos_thetamax();

    Result result;
    for (real_type energy : {50, 100, 200, 1000, 13000})
    {
        this->set_inc_particle(pdg::electron(), MevEnergy{energy});

        WentzelHelper helper(this->particle_track(),
                             material,
                             target_z,
                             wentzel->host_ref(),
                             model_->host_ref().ids,
                             cutoff);

        EXPECT_SOFT_EQ(1.1682, helper.mott_factor());
        result.screen_z.push_back(helper.screening_coefficient());
        // Scale the xs factor by 1 / r_e^2 so the values will be large enough
        // for the soft equivalence comparison to catch any differences
        result.scaled_kin_factor.push_back(helper.kin_factor()
                                           / ipow<2>(constants::r_electron));
        result.cos_thetamax_elec.push_back(helper.cos_thetamax_electron());
        real_type const cos_thetamax_nuc = helper.cos_thetamax_nuclear();
        result.cos_thetamax_nuc.push_back(cos_thetamax_nuc);
        result.xs_elec.push_back(
            helper.calc_xs_electron(cos_thetamax_nuc, cos_thetamax)
            / units::barn);
        result.xs_nuc.push_back(
            helper.calc_xs_nuclear(cos_thetamax_nuc, cos_thetamax)
            / units::barn);
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
    static double const expected_cos_thetamax_elec[] = {0.99989885103277,
                                                        0.99997458240728,
                                                        0.99999362912075,
                                                        0.99999974463379,
                                                        0.99999999848823};
    static double const expected_cos_thetamax_nuc[] = {1, 1, 1, 1, 1};
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
    EXPECT_VEC_SOFT_EQ(expected_screen_z, result.screen_z);
    EXPECT_VEC_SOFT_EQ(expected_scaled_kin_factor, result.scaled_kin_factor);
    EXPECT_VEC_SOFT_EQ(expected_cos_thetamax_elec, result.cos_thetamax_elec);
    EXPECT_VEC_SOFT_EQ(expected_cos_thetamax_nuc, result.cos_thetamax_nuc);
    EXPECT_VEC_SOFT_EQ(expected_xs_elec, result.xs_elec);
    EXPECT_VEC_SOFT_EQ(expected_xs_nuc, result.xs_nuc);
}

TEST_F(CoulombScatteringTest, mott_ratio)
{
    auto wentzel = this->make_wentzel_params();

    static real_type const cos_theta[]
        = {1, 0.9, 0.5, 0.21, 0, -0.1, -0.6, -0.7, -0.9, -1};
    {
        // Test Mott ratios for electrons
        MottElementData::MottCoeffMatrix const& coeffs
            = wentzel->host_ref().mott_coeffs[el_id_].electron;
        MottRatioCalculator calc_mott_ratio(
            coeffs, sqrt(this->particle_track().beta_sq()));

        std::vector<real_type> ratios;
        for (real_type cos_t : cos_theta)
        {
            ratios.push_back(calc_mott_ratio(cos_t));
        }
        static real_type const expected_ratios[] = {
            0.99997507022045,
            1.090740570075,
            0.98638178782896,
            0.83702240402998,
            0.71099171311683,
            0.64712379625713,
            0.30071752615308,
            0.22722448378001,
            0.07702815350459,
            0.00051427465924958,
        };
        EXPECT_VEC_SOFT_EQ(ratios, expected_ratios);
    }
    {
        // Test Mott ratios for positrons
        MottElementData::MottCoeffMatrix const& coeffs
            = wentzel->host_ref().mott_coeffs[el_id_].positron;
        MottRatioCalculator calc_mott_ratio(
            coeffs, sqrt(this->particle_track().beta_sq()));

        std::vector<real_type> ratios;
        for (real_type cos_t : cos_theta)
        {
            ratios.push_back(calc_mott_ratio(cos_t));
        }
        static double const expected_ratios[] = {
            0.99999249638442,
            0.86228266918504,
            0.63153899926215,
            0.49679913349546,
            0.40508196203984,
            0.36255112618068,
            0.15753302403326,
            0.11771390807236,
            0.039017331954949,
            0.00010139510205454,
        };
        EXPECT_VEC_SOFT_EQ(ratios, expected_ratios);
    }
}

TEST_F(CoulombScatteringTest, wokvi_transport_xs)
{
    auto wentzel = this->make_wentzel_params();

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
                             wentzel->host_ref(),
                             model_->host_ref().ids,
                             cutoff);
        WentzelTransportXsCalculator calc_transport_xs(particle, helper);

        for (real_type cos_thetamax : {-1.0, -0.5, 0.0, 0.5, 0.75, 0.99, 1.0})
        {
            // Get cross section in barns
            xs.push_back(calc_transport_xs(cos_thetamax) / units::barn);
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
    int const num_samples = 8;

    auto const material = this->material_track().make_material_view();
    IsotopeView const isotope
        = material.make_element_view(elcomp_id_).make_isotope_view(isocomp_id_);
    auto cutoffs = this->cutoff_params()->get(mat_id_);

    auto& rng_engine = this->rng();

    // Create one params for each form factor
    std::vector<SPWentzel> all_wentzel;
    std::vector<std::string> ff_str;
    for (auto ff : range(NuclearFormFactorType::size_))
    {
        all_wentzel.push_back(this->make_wentzel_params(ff));
        ff_str.push_back(to_cstring(ff));
    }

    std::vector<real_type> cos_theta;
    std::vector<real_type> eloss_frac;

    for (auto particle : {pdg::electron(), pdg::positron()})
    {
        for (auto energy : {0.05, 0.1, 0.5, 1.0, 10.0})
        {
            this->set_inc_particle(particle, MevEnergy{energy});
            for (auto i : range(all_wentzel.size()))
            {
                CoulombScatteringInteractor interact(model_->host_ref(),
                                                     all_wentzel[i]->host_ref(),
                                                     this->particle_track(),
                                                     this->direction(),
                                                     material,
                                                     isotope,
                                                     el_id_,
                                                     cutoffs);

                real_type accum_costheta{0};
                real_type accum_eloss{0};
                for ([[maybe_unused]] int i : range(num_samples))
                {
                    Interaction result = interact(rng_engine);
                    SCOPED_TRACE(result);
                    this->sanity_check(result);

                    real_type ct
                        = dot_product(this->direction(), result.direction);
                    real_type eloss = 1 - result.energy.value() / energy;
                    accum_costheta += ct;
                    accum_eloss += eloss;
                }
                cos_theta.push_back(accum_costheta
                                    * (real_type{1} / num_samples));
                eloss_frac.push_back(accum_eloss
                                     * (real_type{1} / num_samples));
            }
        }
    }

    static double const expected_cos_theta[] = {
        0.99085547184471, 0.99582869481847, 0.99297651186022, 0.96857083475692,
        0.99906508938769, 0.99968090574406, 0.99874182275461, 0.99967220230996,
        0.99954294565515, 0.99955207355357, 0.99916677835997, 0.99983036797864,
        0.99967679868941, 0.99977009860159, 0.99993363923308, 0.99888270406916,
        0.99999832157267, 0.99999216130374, 0.99997210766364, 0.99999516125025,
        0.96007916247231, 0.99752140412521, 0.93688613378615, 0.97878580321185,
        0.99788416453335, 0.99693223370548, 0.99470224997061, 0.99929666580995,
        0.99918098814891, 0.99966885851163, 0.99979645194886, 0.99961272152545,
        0.99990659880816, 0.99984360657577, 0.99983640235532, 0.99987797564342,
        0.99999877815323, 0.99998936058472, 0.99999812696281, 0.99999406093449,
    };
    static double const expected_eloss_frac[] = {
        1.672323716867e-07,  7.6283596037174e-08, 1.2844344925156e-07,
        5.7476665826783e-07, 1.7894823539644e-08, 6.1076806667781e-09,
        2.4082365740474e-08, 6.274270297757e-09,  1.1867170246482e-08,
        1.1630169369226e-08, 2.1634151442518e-08, 4.4044043551184e-09,
        1.1148568179742e-08, 7.9302630850986e-09, 2.2890611212079e-09,
        3.8540216684702e-08, 3.1559359292554e-10, 1.4739048453949e-09,
        5.2445774251231e-09, 9.0982694989084e-10, 7.3005844924978e-07,
        4.5327830003328e-08, 1.1542025986633e-06, 3.8795855997198e-07,
        4.049852457555e-08,  5.8719126158624e-08, 1.0140249237511e-07,
        1.3462293807498e-08, 2.1265202584941e-08, 8.5979108832568e-09,
        5.2850158138718e-09, 1.0055477520088e-08, 3.2217987722127e-09,
        5.394665020253e-09,  5.6431687378566e-09, 4.2091317237203e-09,
        2.2974308566681e-10, 2.0005222339803e-09, 3.5218594707231e-10,
        1.1167185509642e-09,
    };
    EXPECT_VEC_SOFT_EQ(expected_cos_theta, cos_theta);
    EXPECT_VEC_SOFT_EQ(expected_eloss_frac, eloss_frac);
}

TEST_F(CoulombScatteringTest, distribution)
{
    auto wentzel = this->make_wentzel_params();

    auto const material = this->material_track().make_material_view();
    IsotopeView const isotope
        = material.make_element_view(elcomp_id_).make_isotope_view(isocomp_id_);

    // TODO: Use proton ParticleId{2}
    MevEnergy const cutoff
        = this->cutoff_params()->get(mat_id_).energy(ParticleId{0});

    std::vector<real_type> avg_angles;
    std::vector<real_type> avg_engine_samples;

    for (auto pdg : {pdg::electron(), pdg::positron()})
    {
        for (real_type energy : {1, 50, 100, 200, 1000, 13000})
        {
            this->set_inc_particle(pdg, MevEnergy{energy});

            WentzelHelper helper(this->particle_track(),
                                 material,
                                 isotope.atomic_number(),
                                 wentzel->host_ref(),
                                 model_->host_ref().ids,
                                 cutoff);
            WentzelDistribution sample_angle(wentzel->host_ref(),
                                             helper,
                                             this->particle_track(),
                                             isotope,
                                             el_id_,
                                             helper.cos_thetamax_nuclear(),
                                             model_->host_ref().cos_thetamax());

            RandomEngine& rng = this->rng();

            real_type avg_angle = 0;

            int const num_samples = 4096;
            for ([[maybe_unused]] int i : range(num_samples))
            {
                avg_angle += sample_angle(rng);
            }

            avg_angle /= num_samples;
            avg_angles.push_back(avg_angle);
            avg_engine_samples.push_back(real_type(rng.count()) / num_samples);
        }
    }

    static double const expected_avg_angles[] = {
        0.99957853627426,
        0.99999954645904,
        0.99999989882947,
        0.99999996985799,
        0.99999999945722,
        0.99999999999487,
        0.99970212785622,
        0.99999969317473,
        0.99999989582094,
        0.99999998024112,
        0.99999999932915,
        0.99999999996876,
    };
    static double const expected_avg_engine_samples[] = {
        5.9287109375,
        5.93359375,
        5.9287109375,
        5.943359375,
        5.927734375,
        5.927734375,
        5.93701171875,
        5.9306640625,
        5.9345703125,
        5.92431640625,
        5.9345703125,
        5.9267578125,
    };
    EXPECT_VEC_SOFT_EQ(expected_avg_angles, avg_angles);
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
