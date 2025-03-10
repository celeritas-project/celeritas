//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
        return std::make_shared<WentzelOKVIParams>(
            this->material_params(), this->particle_params(), options);
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

    auto const material = this->material_track().material_record();
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

    static double const expected_screen_z[] = {
        2.1181757502465e-08,
        5.3641196710457e-09,
        1.3498490873627e-09,
        5.4280909096648e-11,
        3.2158426877075e-13,
    };
    static double const expected_scaled_kin_factor[] = {
        0.018652406025343,
        0.0047099158443663,
        0.0011834423731331,
        4.7530717147601e-05,
        2.8151207657337e-07,
    };
    static double const expected_cos_thetamax_elec[] = {
        0.99989885103277,
        0.99997458240728,
        0.99999362912075,
        0.99999974463379,
        0.99999999848823,
    };
    static double const expected_cos_thetamax_nuc[] = {1, 1, 1, 1, 1};
    static double const expected_xs_elec[] = {
        40826.467546089,
        40708.229241237,
        40647.018240347,
        40596.954587654,
        40585.256749842,
    };
    static double const expected_xs_nuc[] = {
        1184463.4066054,
        1181036.9225682,
        1179263.035472,
        1177812.1841967,
        1177473.1776044,
    };
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

    auto const material = this->material_track().material_record();
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
    static double const expected_xs[] = {
        0.18738907038684,
        0.18698029572191,
        0.18529403118945,
        0.1804875301691,
        0.17432529841181,
        0.14071448257828,
        0,
        0.050844259181328,
        0.05074156113331,
        0.050317873132892,
        0.049110175331928,
        0.047561821665737,
        0.039116563913081,
        0,
        0.0023937925545266,
        0.0023896680528842,
        0.0023726515988718,
        0.0023241468787067,
        0.0022619602763042,
        0.0019227725532218,
        0,
        3.4052045441207e-07,
        3.4010758776621e-07,
        3.3840422185375e-07,
        3.3354884426624e-07,
        3.2732389416414e-07,
        2.9337081291625e-07,
        0,
        3.9098094206748e-09,
        3.9056807162416e-09,
        3.8886469004431e-09,
        3.8400926779737e-09,
        3.7778426043855e-09,
        3.4383086689208e-09,
        0,
    };
    EXPECT_VEC_SOFT_EQ(expected_xs, xs);
}

TEST_F(CoulombScatteringTest, simple_scattering)
{
    int const num_samples = 1024;

    auto const material = this->material_track().material_record();
    IsotopeView const isotope
        = material.element_record(elcomp_id_).isotope_record(isocomp_id_);
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
        for (auto log_energy : range(-4, 6).step(2))
        {
            real_type energy = std::pow(real_type{10}, log_energy);
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
                for (int j : range(num_samples))
                {
                    Interaction result;
                    try
                    {
                        result = interact(rng_engine);
                    }
                    catch (DebugError const& e)
                    {
                        ADD_FAILURE() << e.what() << ": for PDG "
                                      << particle.get() << ", E=" << energy
                                      << "MeV, form factor = " << ff_str[i];
                        continue;
                    }
                    if (j < 8)
                    {
                        SCOPED_TRACE(result);
                        this->sanity_check(result);
                    }

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
        0.30959693518702, 0.30362801736357, 0.34400141555852, 0.27468879706731,
        0.96386966095849, 0.96315701193745, 0.95385031468713, 0.96058261257009,
        0.99947380591936, 0.99903504822288, 0.99955489342985, 0.99962786618013,
        0.99999941717203, 0.99999990332905, 0.99999995040553, 0.99999993537485,
        0.99999999999292, 0.99999999999046, 0.99999999999287, 0.9999999999951,
        0.21467076189019, 0.22333217702059, 0.19782535179014, 0.19180477684422,
        0.95348944411602, 0.95231010283708, 0.95499925870039, 0.95810237285708,
        0.99972525491626, 0.99947505610692, 0.99967163248489, 0.99969994907305,
        0.99999992659793, 0.99999994637187, 0.99999993733887, 0.99999995223927,
        0.99999999995776, 0.99999999999134, 0.99999999999354, 0.99999999998984,
    };
    static double const expected_eloss_frac[] = {
        1.2038042397826e-05, 1.2142117225683e-05, 1.1438161896383e-05,
        1.264670859096e-05,  6.3608414025622e-07, 6.4863054261043e-07,
        8.1247582681175e-07, 6.9395307116373e-07, 1.815056819112e-08,
        3.3285038144324e-08, 1.5353566031615e-08, 1.2836452633258e-08,
        1.0044336467164e-09, 1.6660085756705e-10, 8.5470164257716e-11,
        1.113737352126e-10,  1.2077655498974e-12, 1.627270800747e-12,
        1.2157304243171e-12, 8.3538709803876e-13, 1.3693201535358e-05,
        1.3542181923804e-05, 1.3986924425419e-05, 1.4091901422063e-05,
        8.1883052876781e-07, 8.395920386762e-07,  7.9225056037789e-07,
        7.3761848967387e-07, 9.4771081645328e-09, 1.810747388164e-08,
        1.1326769579559e-08, 1.035001138234e-08,  1.2649970914519e-10,
        9.2421692261119e-11, 1.0798899632587e-10, 8.230992927169e-11,
        7.2070335810359e-12, 1.4779506828447e-12, 1.1019583683047e-12,
        1.7337795695654e-12,
    };

    EXPECT_VEC_SOFT_EQ(expected_cos_theta, cos_theta);
    EXPECT_VEC_SOFT_EQ(expected_eloss_frac, eloss_frac);
}

TEST_F(CoulombScatteringTest, distribution)
{
    auto wentzel = this->make_wentzel_params();

    auto const material = this->material_track().material_record();
    IsotopeView const isotope
        = material.element_record(elcomp_id_).isotope_record(isocomp_id_);

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
