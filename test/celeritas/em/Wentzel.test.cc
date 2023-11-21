//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/Wentzel.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/interactor/WentzelInteractor.hh"
#include "celeritas/em/model/WentzelModel.hh"
#include "celeritas/em/process/CoulombScatteringProcess.hh"
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
        using namespace constants;
        // Need to include protons
        constexpr units::MevMass emass{0.5109989461};
        ParticleParams::Input par_inp
            = {{"electron",
                pdg::electron(),
                emass,
                celeritas::units::ElementaryCharge{-1},
                stable_decay_constant},
               {"positron",
                pdg::positron(),
                emass,
                celeritas::units::ElementaryCharge{1},
                stable_decay_constant},
               {"proton",
                pdg::proton(),
                units::MevMass{938.28},
                celeritas::units::ElementaryCharge{1},
                stable_decay_constant}};
        this->set_particle_params(std::move(par_inp));

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
                {},
                ImportProcessClass::coulomb_scat,
                {ImportModelClass::e_coulomb_scattering});
            ImportProcess ip_positron = ip_electron;
            ip_positron.particle_pdg = pdg::positron().get();
            this->set_imported_processes(
                {std::move(ip_electron), std::move(ip_positron)});
        }

        // Use default options
        WentzelModel::Options options;

        model_ = std::make_shared<WentzelModel>(ActionId{0},
                                                *this->particle_params(),
                                                *this->material_params(),
                                                options,
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
    std::shared_ptr<WentzelModel> model_;
};

TEST_F(CoulombScatteringTest, wokvi_xs)
{
    WentzelHostRef const& data = model_->host_ref();

    AtomicNumber const target_z
        = this->material_params()->get(ElementId{0}).atomic_number();

    const real_type cutoff_energy = value_as<units::MevEnergy>(
        this->cutoff_params()
            ->get(MaterialId{0})
            .energy(this->particle_track().particle_id()));

    const std::vector<real_type> energies = {50, 100, 200, 1000, 13000};

    static real_type const expected_screen_z[] = {2.1181757502465e-08,
                                                  5.3641196710457e-09,
                                                  1.3498490873627e-09,
                                                  5.4280909096648e-11,
                                                  3.2158426877075e-13};

    static real_type const expected_cos_t_max[] = {0.99989885103277,
                                                   0.99997458240728,
                                                   0.99999362912075,
                                                   0.99999974463379,
                                                   0.99999999848823};

    static real_type const expected_xsecs[] = {0.033319844069031,
                                               0.033319738720425,
                                               0.033319684608429,
                                               0.033319640583261,
                                               0.03331963032739};

    std::vector<real_type> xsecs, cos_t_maxs, screen_zs;
    for (real_type energy : energies)
    {
        this->set_inc_particle(pdg::electron(), MevEnergy{energy});

        WentzelRatioCalculator calc(
            particle_track(), target_z, data, cutoff_energy);

        xsecs.push_back(calc());
        cos_t_maxs.push_back(calc.cos_t_max_elec());
        screen_zs.push_back(calc.screening_coefficient());
    }

    EXPECT_VEC_SOFT_EQ(expected_xsecs, xsecs);
    EXPECT_VEC_SOFT_EQ(expected_screen_z, screen_zs);
    EXPECT_VEC_SOFT_EQ(expected_cos_t_max, cos_t_maxs);
}

TEST_F(CoulombScatteringTest, mott_xs)
{
    WentzelHostRef const& data = model_->host_ref();

    WentzelElementData const& element_data = data.elem_data[ElementId(0)];
    MottRatioCalculator xsec(element_data, sqrt(particle_track().beta_sq()));

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

TEST_F(CoulombScatteringTest, simple_scattering)
{
    int const num_samples = 10;

    static real_type const expected_angle[] = {1,
                                               0.99999999776622,
                                               0.99999999990987,
                                               0.99999999931707,
                                               0.99999999847986,
                                               0.9999999952274,
                                               0.99999999905465,
                                               0.99999999375773,
                                               1,
                                               0.99999999916491};
    static real_type const expected_energy[] = {200,
                                                199.99999999847,
                                                199.99999999994,
                                                199.99999999953,
                                                199.99999999896,
                                                199.99999999673,
                                                199.99999999935,
                                                199.99999999572,
                                                200,
                                                199.99999999943};

    const IsotopeView isotope = this->material_track()
                                    .make_material_view()
                                    .make_element_view(ElementComponentId{0})
                                    .make_isotope_view(IsotopeComponentId{0});
    auto cutoffs = this->cutoff_params()->get(MaterialId{0});

    WentzelInteractor interact(model_->host_ref(),
                               this->particle_track(),
                               this->direction(),
                               isotope,
                               ElementId{0},
                               cutoffs);
    RandomEngine& rng_engine = this->rng();

    std::vector<real_type> angle;
    std::vector<real_type> energy;

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

    static real_type const expected_avg_angles[] = {0.99999962180819,
                                                    0.99999973999034,
                                                    0.9999999728531,
                                                    0.99999999909264,
                                                    0.99999999999393};

    for (size_t i : range(energies.size()))
    {
        this->set_inc_particle(pdg::electron(), MevEnergy{energies[i]});

        WentzelElementData const& element_data = data.elem_data[ElementId(0)];

        const IsotopeView isotope
            = this->material_track()
                  .make_material_view()
                  .make_element_view(ElementComponentId{0})
                  .make_isotope_view(IsotopeComponentId{0});

        const real_type cutoff_energy = value_as<units::MevEnergy>(
            this->cutoff_params()
                ->get(MaterialId{0})
                .energy(ParticleId{0}));  // TODO: Use proton ParticleId{2}

        WentzelDistribution distrib(
            particle_track(), isotope, element_data, cutoff_energy, data);

        RandomEngine& rng_engine = this->rng();

        real_type avg_angle = 0;

        int const num_samples = 4096;
        for ([[maybe_unused]] int i : range(num_samples))
        {
            avg_angle += distrib(rng_engine);
        }

        avg_angle /= num_samples;

        EXPECT_SOFT_NEAR(
            expected_avg_angles[i], avg_angle, std::sqrt(num_samples));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
