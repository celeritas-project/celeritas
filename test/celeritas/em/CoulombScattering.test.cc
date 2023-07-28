//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/CoulombScattering.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/interactor/WentzelInteractor.hh"
#include "celeritas/em/model/WentzelModel.hh"
#include "celeritas/em/model/detail/MottInterpolatedCoefficients.hh"
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
        // Set up shared material data
        // TODO: Use multiple elements to test elements are picked correctly
        MaterialParams::Input mat_inp;
        mat_inp.isotopes = {{AtomicNumber{29},
                             AtomicNumber{63},
                             units::MevMass{58618.5},
                             "63Cu"},
                            {AtomicNumber{29},
                             AtomicNumber{65},
                             units::MevMass{60479.8},
                             "65Cu"}};
        mat_inp.elements = {{AtomicNumber{29},
                             units::AmuMass{63.546},
                             {{IsotopeId{0}, 0.692}, {IsotopeId{1}, 0.308}},
                             "Cu"}};
        mat_inp.materials = {
            {0.141 * constants::na_avogadro,
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
                pdg::electron(),  // TODO: no secondary?
                ImportProcessClass::coulomb_scat,
                {ImportModelClass::e_coulomb_scattering});
            ImportProcess ip_positron = ip_electron;
            ip_positron.particle_pdg = pdg::positron().get();
            this->set_imported_processes(
                {std::move(ip_electron), std::move(ip_positron)});
        }

        model_ = std::make_shared<WentzelModel>(ActionId{0},
                                                *this->particle_params(),
                                                *this->material_params(),
                                                ImportEmParameters{},
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
        EXPECT_GT(this->particle_track().energy().value(),
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
    std::shared_ptr<WentzelModel> model_;
};

TEST_F(CoulombScatteringTest, wokvi_data)
{
    WentzelHostRef const& data = model_->host_ref();

    // Check element data is filled in correctly
    unsigned int const num_elements = this->material_params()->num_elements();
    for (auto el_id : range(ElementId(num_elements)))
    {
        int const z = this->material_params()->get(el_id).atomic_number().get();
        int const mott_index = (z <= 92) ? z : 0;

        WentzelElementData const& element_data = data.elem_data[el_id];
        for (auto i : range(5))
        {
            for (auto j : range(6))
            {
                EXPECT_EQ(detail::interpolated_mott_coeffs[mott_index][i][j],
                          element_data.mott_coeff[i][j]);
            }
        }
    }
}

TEST_F(CoulombScatteringTest, wokvi_xs)
{
    AtomicNumber const target_z
        = this->material_params()->get(ElementId(0)).atomic_number();

    static double const cos_ts[] = {1, 0.8, 0.3, 0, -0.4, -0.7, -1};
    static double const screenings[] = {1, 1.13, 1.73, 2.5};
    static double const expected_xsecs[] = {0,
                                            0,
                                            0,
                                            0,
                                            0.0062305295950156,
                                            0.0059359585318953,
                                            0.005117822394691,
                                            0.0046204620462046,
                                            0.017565872020075,
                                            0.017072975232163,
                                            0.015593508008911,
                                            0.014605067064083,
                                            0.02247191011236,
                                            0.02203372297507,
                                            0.020670856364049,
                                            0.019718309859155,
                                            0.027613412228797,
                                            0.027327211744653,
                                            0.026401956314502,
                                            0.025721784776903,
                                            0.030713640469738,
                                            0.030567022057892,
                                            0.030081474711727,
                                            0.029712858926342,
                                            0.033333333333333,
                                            0.033333333333333,
                                            0.033333333333333,
                                            0.033333333333333};

    std::vector<double> xsecs;
    for (double cos_t : cos_ts)
    {
        for (double screening : screenings)
        {
            WentzelXsCalculator xsec(target_z, screening, cos_t);
            xsecs.push_back(xsec());
        }
    }

    EXPECT_VEC_SOFT_EQ(expected_xsecs, xsecs);
}

TEST_F(CoulombScatteringTest, mott_xs)
{
    WentzelHostRef const& data = model_->host_ref();

    real_type inc_energy
        = value_as<units::MevEnergy>(particle_track().energy());
    real_type inc_mass = value_as<units::MevMass>(particle_track().mass());
    WentzelElementData const& element_data = data.elem_data[ElementId(0)];

    MottXsCalculator xsec(element_data, inc_energy, inc_mass);

    static double const cos_ts[]
        = {1, 0.9, 0.5, 0.21, 0, -0.1, -0.6, -0.7, -0.9, -1};
    static double const expected_xsecs[] = {0.99997507022045,
                                            1.090740570075,
                                            0.98638178782896,
                                            0.83702240402998,
                                            0.71099171311683,
                                            0.64712379625713,
                                            0.30071752615308,
                                            0.22722448378001,
                                            0.07702815350459,
                                            0.00051427465924958};

    std::vector<double> xsecs;
    for (double cos_t : cos_ts)
    {
        xsecs.push_back(xsec(cos_t));
    }

    EXPECT_VEC_SOFT_EQ(xsecs, expected_xsecs);
}

TEST_F(CoulombScatteringTest, simple_scattering)
{
    int const num_samples = 4;

    static double const expected_angle[] = {
        0.99999999991325, 0.99999999998064, 0.9999999781261, 0.99999999847986};
    static double const expected_energy[]
        = {199.99999999994, 199.99999999999, 199.99999998546, 199.99999999896};

    auto material_view = this->material_track().make_material_view();
    auto cutoffs = this->cutoff_params()->get(MaterialId{0});

    WentzelInteractor interact(model_->host_ref(),
                               this->particle_track(),
                               this->direction(),
                               material_view,
                               ElementComponentId{0},
                               cutoffs);
    RandomEngine& rng_engine = this->rng();

    std::vector<double> angle;
    std::vector<double> energy;

    for ([[maybe_unused]] int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        energy.push_back(result.energy.value());
        angle.push_back(dot_product(this->direction(), result.direction));
    }

    EXPECT_VEC_SOFT_EQ(expected_angle, angle);
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
}

TEST_F(CoulombScatteringTest, distribution)
{
    WentzelHostRef const& data = model_->host_ref();

    const std::vector<real_type> energies = {50, 100, 200, 1000, 13000};

    std::vector<double> screen_zs, cos_t_maxs;

    static double const expected_avg_angles[] = {0.99999962180819,
                                                 0.99999973999034,
                                                 0.9999999728531,
                                                 0.99999999909264,
                                                 0.99999999999393};
    static double const expected_screen_zs[] = {2.1181823391965e-08,
                                                5.3641363570627e-09,
                                                1.3498532863005e-09,
                                                5.4281077946754e-11,
                                                3.2158526911394e-13};

    static double const expected_cos_t_max[] = {0.99989885103277,
                                                0.99997458240728,
                                                0.99999362912075,
                                                0.99999974463379,
                                                0.99999999848823};

    for (size_t i : range(energies.size()))
    {
        const real_type inc_energy = energies[i];
        const real_type inc_mass
            = value_as<units::MevMass>(this->particle_track().mass());
        const IsotopeView isotope
            = this->material_track()
                  .make_material_view()
                  .make_element_view(ElementComponentId{0})
                  .make_isotope_view(IsotopeComponentId{0});
        WentzelElementData const& element_data = data.elem_data[ElementId(0)];
        const real_type cutoff_energy = value_as<units::MevEnergy>(
            this->cutoff_params()
                ->get(MaterialId{0})
                .energy(this->particle_track().particle_id()));

        WentzelDistribution distrib(inc_energy,
                                    inc_mass,
                                    isotope,
                                    element_data,
                                    cutoff_energy,
                                    true,
                                    data);

        screen_zs.push_back(distrib.compute_screening_coefficient());
        cos_t_maxs.push_back(distrib.compute_max_electron_cos_t());

        RandomEngine& rng_engine = this->rng();

        double avg_angle = 0;
        double avg_x = 0;
        double avg_y = 0;

        int const num_samples = 4096;
        for ([[maybe_unused]] int i : range(num_samples))
        {
            Real3 dir = distrib(rng_engine);
            avg_x += dir[0];
            avg_y += dir[1];
            avg_angle += dir[2];
        }

        avg_x /= num_samples;
        avg_y /= num_samples;
        avg_angle /= num_samples;

        EXPECT_SOFT_NEAR(0, avg_x, std::sqrt(num_samples));
        EXPECT_SOFT_NEAR(0, avg_y, std::sqrt(num_samples));
        EXPECT_SOFT_NEAR(
            expected_avg_angles[i], avg_angle, std::sqrt(num_samples));
    }

    EXPECT_VEC_SOFT_EQ(expected_screen_zs, screen_zs);
    EXPECT_VEC_SOFT_EQ(expected_cos_t_max, cos_t_maxs);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
