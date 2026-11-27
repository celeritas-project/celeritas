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
                {ImportModelClass::e_coulomb_scattering},
                {{0, 1e12}});
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

    SPWentzel make_wentzel_params(
        NuclearFormFactorType ff = NuclearFormFactorType::exponential)
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
    PhysMatId mat_id_{0};
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
        result.scaled_kin_factor.push_back(
            helper.kin_factor() / ipow<2>(constants::r_electron));
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

// Deflections near zero can't be represented as cosines in single precision
TEST_F(CoulombScatteringTest, TEST_IF_CELERITAS_DOUBLE(cos_thetamax_electron))
{
    auto wentzel = this->make_wentzel_params();

    auto const material = this->material_track().material_record();
    AtomicNumber const target_z
        = this->material_params()->get(el_id_).atomic_number();
    MevEnergy const cutoff
        = this->cutoff_params()->get(mat_id_).energy(ParticleId{0});

    // Calculate the maximum deflection 1 - cos(theta) off electrons, and
    // the cross section for scattering off electrons over all angles
    std::vector<real_type> deflection;
    std::vector<real_type> xs_elec;
    auto calc = [&](PDGNumber pdg, real_type energy) {
        this->set_inc_particle(pdg, MevEnergy{energy});
        WentzelHelper helper(this->particle_track(),
                             material,
                             target_z,
                             wentzel->host_ref(),
                             model_->host_ref().ids,
                             cutoff);
        deflection.push_back(1 - helper.cos_thetamax_electron());
        xs_elec.push_back(helper.calc_xs_electron(1, -1) / units::barn);
    };

    // Reference values from G4WentzelOKandVIxSection with a 0.5 MeV cut,
    // calculated as min(cut, tmax) m_e / p^2 to avoid cancellation. The
    // tolerance accounts for recovering a small deflection from a cosine.
    for (real_type energy : {10, 100, 1000, 13000})
    {
        calc(pdg::proton(), energy);
    }
    static double const expected_proton[] = {
        5.925529089626828e-07,
        5.92491074377484e-07,
        8.882118678212865e-08,
        1.3211256916404578e-09,
    };
    EXPECT_VEC_NEAR(expected_proton, deflection, 1e-6_r);

    // No scattering off electrons if the outgoing positron would have no
    // kinetic energy
    deflection.clear();
    xs_elec.clear();
    for (real_type energy : {0.25, 0.5, 1.0, 10.0})
    {
        calc(pdg::positron(), energy);
    }
    static double const expected_positron[] = {
        0.0,
        0.0,
        0.18497988777806718,
        0.0024311964170518063,
    };
    EXPECT_VEC_SOFT_EQ(expected_positron, deflection);
    EXPECT_EQ(0, xs_elec[0]);
    EXPECT_EQ(0, xs_elec[1]);
    EXPECT_LT(0, xs_elec[2]);
    EXPECT_LT(0, xs_elec[3]);
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
            real_type energy = std::pow(10_r, log_energy);
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
                cos_theta.push_back(accum_costheta * (1_r / num_samples));
                eloss_frac.push_back(accum_eloss * (1_r / num_samples));
            }
        }
    }

    static double const expected_cos_theta[] = {
        0.285268482356402, 0.319111684990326, 0.298354772775234,
        0.33040392929038,  0.969289079711668, 0.955453483995664,
        0.964579080727355, 0.958001523415026, 0.999798579638989,
        0.999510311148769, 0.999783719772236, 0.999752058579047,
        0.999999929733473, 0.999999953732833, 0.999999959343023,
        0.999999933380838, 0.999999999990511, 0.99999999999046,
        0.99999999999287,  0.9999999999951,   0.160104647243097,
        0.203720958026138, 0.188922436206288, 0.191678723654579,
        0.958461810664268, 0.952200053588387, 0.954023677085216,
        0.953979988606363, 0.999592679710791, 0.999673123654627,
        0.99980821074142,  0.999665306247582, 0.999999917655697,
        0.999998167033391, 0.99999991947531,  0.999999877495995,
        0.999999999990475, 0.9999999999941,   0.99999999999354,
        0.99999999998984,
    };
    static double const expected_eloss_frac[] = {
        1.24622392637623e-05, 1.1872142976693e-05,  1.22340662779997e-05,
        1.16752469390891e-05, 5.40674375022884e-07, 7.84251686915326e-07,
        6.23594227307711e-07, 7.39393867211775e-07, 6.94783250159911e-09,
        1.68914074870309e-08, 7.4604110298325e-09,  8.55253769931411e-09,
        1.21095981115912e-10, 7.97359451685728e-11, 7.00674530331943e-11,
        1.14810182628294e-10, 1.61886519814414e-12, 1.59694111233344e-12,
        1.37848445765781e-12, 7.49713008688091e-13, 1.46446265678251e-05,
        1.38841277612525e-05, 1.41421563243938e-05, 1.40940988240966e-05,
        7.31291133991606e-07, 8.41529675028895e-07, 8.09425517892242e-07,
        8.10193980857812e-07, 1.40501713227206e-08, 1.12753314856487e-08,
        6.61561476585602e-09, 1.15449868033635e-08, 1.41910588965719e-10,
        3.15889028736777e-09, 1.38774707654152e-10, 2.1112103862006e-10,
        1.62498866359412e-12, 1.00654261401406e-12, 1.13231104180433e-12,
        1.61846881382988e-12,
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
        0.999586320451099,
        0.99999975394707,
        0.999999842187494,
        0.999999974011507,
        0.999999999368774,
        0.999999999996155,
        0.999592273837762,
        0.999999586207064,
        0.999999848966014,
        0.999999978902978,
        0.999999999271879,
        0.999999999996316,
    };
    static double const expected_avg_engine_samples[] = {
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
    };
    EXPECT_VEC_SOFT_EQ(expected_avg_angles, avg_angles);
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
