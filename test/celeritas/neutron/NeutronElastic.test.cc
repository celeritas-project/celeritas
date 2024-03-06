//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/NeutronElastic.test.cc
//---------------------------------------------------------------------------//
#include <memory>

#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/io/NeutronXsReader.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/neutron/NeutronTestBase.hh"
#include "celeritas/neutron/interactor/ChipsNeutronElasticInteractor.hh"
#include "celeritas/neutron/model/ChipsNeutronElasticModel.hh"
#include "celeritas/neutron/xs/NeutronElasticMacroXsCalculator.hh"
#include "celeritas/neutron/xs/NeutronElasticMicroXsCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class NeutronElasticTest : public NeutronTestBase
{
  protected:
    using MevEnergy = units::MevEnergy;
    using SPConstNElasticModel
        = std::shared_ptr<ChipsNeutronElasticModel const>;

    void SetUp() override
    {
        using namespace units;

        // Load neutron elastic cross section data
        std::string data_path = this->test_data_path("celeritas", "");
        NeutronXsReader read_el_data(data_path.c_str());

        // Set up the default particle: 100 MeV neutron along +z direction
        auto const& particles = *this->particle_params();
        this->set_inc_particle(pdg::neutron(), MevEnergy{100});
        this->set_inc_direction({0, 0, 1});

        // Set up the default material
        this->set_material("HeCu");
        model_ = std::make_shared<ChipsNeutronElasticModel>(
            ActionId{0}, particles, *this->material_params(), read_el_data);
    }

  protected:
    SPConstNElasticModel model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(NeutronElasticTest, micro_xs)
{
    // Calculate the elastic neutron-nucleus microscopic cross section in the
    // valid range [1e-5:2e+4] (MeV)
    using XsCalculator = NeutronElasticMicroXsCalculator;

    // Set the target element: Cu
    ElementId el_id{1};

    // Check the size of the element cross section data (G4PARTICLEXS4.0)
    NeutronElasticRef shared = model_->host_ref();
    GenericGridData grid = shared.micro_xs[el_id];
    EXPECT_EQ(grid.grid.size(), 181);

    // Microscopic cross section (\f$ mm^{2} \f$) in [1e-05:1e+4] (MeV)
    std::vector<real_type> const expected_micro_xs = {7.77548e-24,
                                                      7.54911e-24,
                                                      5.5795e-24,
                                                      1.52568e-23,
                                                      6.55421e-24,
                                                      3.32152e-24,
                                                      1.89914e-24,
                                                      1.16444e-24,
                                                      4.78256e-25,
                                                      5.04635e-25};

    real_type energy = 1e-5;
    real_type const factor = 1e+1;
    for (auto i : range(expected_micro_xs.size()))
    {
        XsCalculator calc_micro_xs(shared, MevEnergy{energy});
        EXPECT_SOFT_EQ(calc_micro_xs(el_id), expected_micro_xs[i]);
        energy *= factor;
    }

    // Check the elastic cross section at the upper bound (20 GeV)
    XsCalculator calc_upper_xs(shared, MevEnergy{2e+4});
    EXPECT_SOFT_EQ(calc_upper_xs(el_id), 4.67e-23);
}

TEST_F(NeutronElasticTest, macro_xs)
{
    // Calculate the CHIPS elastic neutron-nucleus macroscopic cross section
    // (\f$ cm^{-1} \f$) in the valid range [1e-5:2000] (MeV)
    auto material = this->material_track().make_material_view();
    NeutronElasticMacroXsCalculator calc_xs(model_->host_ref(), material);

    std::vector<real_type> const expected_macro_xs = {0.54527696304096029,
                                                      0.52957250425960667,
                                                      0.39293046559628597,
                                                      1.064289155863954,
                                                      0.46061008089551958,
                                                      0.28363890447070578,
                                                      0.1425404724303469,
                                                      0.081439808202180136,
                                                      0.033470862079907314};

    real_type energy = 1e-5;
    real_type const factor = 1e+1;
    for (auto i : range(expected_macro_xs.size()))
    {
        EXPECT_SOFT_EQ(
            native_value_to<units::InvCmXs>(calc_xs(MevEnergy{energy})).value(),
            expected_macro_xs[i]);
        energy *= factor;
    }

    // Check the CHIPS macroscopic cross section at the upper bound (20 GeV)
    EXPECT_SOFT_EQ(
        native_value_to<units::InvCmXs>(calc_xs(MevEnergy{2000})).value(),
        0.036279681208164501);
}

TEST_F(NeutronElasticTest, diff_xs_coeffs)
{
    // Get A-dependent parameters of CHIPS differential cross sections used
    // for sampling the momentum transfer.
    auto const& coeffs = model_->host_ref().coeffs;

    // Set the target isotope: He4 (36 parameters for light nuclei, A <= 6)
    ChipsDiffXsCoefficients he4_coeff = coeffs[IsotopeId{1}];
    EXPECT_EQ(he4_coeff.par.size(), 42);
    EXPECT_EQ(he4_coeff.par[0], 16000);
    EXPECT_SOFT_EQ(he4_coeff.par[10], 26.99741289550769);
    EXPECT_SOFT_EQ(he4_coeff.par[20], 0.003);
    EXPECT_SOFT_EQ(he4_coeff.par[30], 731.96468887355979);
    EXPECT_SOFT_EQ(he4_coeff.par[35], 38680.596799999999);
    EXPECT_EQ(he4_coeff.par[36], 0);

    // Set the target isotope: Cu63 (42 parameters for heavy nuclei, A > 6)
    ChipsDiffXsCoefficients cu63_coeff = coeffs[IsotopeId{2}];
    EXPECT_EQ(cu63_coeff.par.size(), 42);
    EXPECT_SOFT_EQ(cu63_coeff.par[0], 527.781478797624);
    EXPECT_SOFT_EQ(cu63_coeff.par[10], 9.842761904761872);
    EXPECT_SOFT_EQ(cu63_coeff.par[20], 4.5646562677427038);
    EXPECT_SOFT_EQ(cu63_coeff.par[30], 1984873.0860000001);
    EXPECT_SOFT_EQ(cu63_coeff.par[35], 0.15874507866387544);
    EXPECT_SOFT_EQ(cu63_coeff.par[41], 7129.2726746278049);
}

TEST_F(NeutronElasticTest, basic)
{
    NeutronElasticRef shared = model_->host_ref();
    RandomEngine& rng_engine = this->rng();

    // Sample neutron-He4 interactions
    IsotopeView isotope_he4 = this->material_track()
                                  .make_material_view()
                                  .make_element_view(ElementComponentId{0})
                                  .make_isotope_view(IsotopeComponentId{1});
    ChipsNeutronElasticInteractor interact_light_target(
        shared, this->particle_track(), this->direction(), isotope_he4);

    // Sample neutron-Cu63 interactions
    IsotopeView isotope_cu63 = this->material_track()
                                   .make_material_view()
                                   .make_element_view(ElementComponentId{1})
                                   .make_isotope_view(IsotopeComponentId{0});
    ChipsNeutronElasticInteractor interact_heavy_target(
        shared, this->particle_track(), this->direction(), isotope_cu63);

    // Produce four samples from the original incident energy/dir
    std::vector<real_type> neutron_energy_he4;
    std::vector<real_type> cos_theta_he4;
    std::vector<real_type> neutron_energy_cu63;
    std::vector<real_type> cos_theta_cu63;

    for ([[maybe_unused]] int i : range(4))
    {
        // Scattering with a light target (A <= 6)
        Interaction result_he4 = interact_light_target(rng_engine);
        neutron_energy_he4.push_back(result_he4.energy.value());
        cos_theta_he4.push_back(
            dot_product(result_he4.direction, this->direction()));

        // Scattering with a heavy target (A > 6)
        Interaction result_cu63 = interact_heavy_target(rng_engine);
        neutron_energy_cu63.push_back(result_cu63.energy.value());
        cos_theta_cu63.push_back(
            dot_product(result_cu63.direction, this->direction()));
    }

    // Note: these are "gold" values based on the host RNG.
    real_type const expected_energy_he4[] = {
        94.1224768910961, 86.8933037406002, 94.6854232398921, 95.9844475596542};

    real_type const expected_energy_cu63[] = {
        99.9721412254339, 99.5806397008464, 99.9885741516896, 99.9799810689728};

    real_type const expected_cos_theta_he4[] = {0.877710038016294,
                                                0.716934339548103,
                                                0.889730991670609,
                                                0.917213465271479};
    real_type const expected_cos_theta_cu63[] = {0.991747504314719,
                                                 0.875520129744385,
                                                 0.996615655776977,
                                                 0.994070112946167};

    EXPECT_VEC_SOFT_EQ(expected_energy_he4, neutron_energy_he4);
    EXPECT_VEC_SOFT_EQ(expected_cos_theta_he4, cos_theta_he4);
    EXPECT_VEC_SOFT_EQ(expected_energy_cu63, neutron_energy_cu63);
    EXPECT_VEC_SOFT_EQ(expected_cos_theta_cu63, cos_theta_cu63);
}

TEST_F(NeutronElasticTest, extended)
{
    // Set the target isotope : Cu63
    ElementComponentId el_id{1};
    IsotopeComponentId iso_id{0};
    IsotopeView const isotope_he4 = this->material_track()
                                        .make_material_view()
                                        .make_element_view(el_id)
                                        .make_isotope_view(iso_id);
    // Sample interaction
    NeutronElasticRef shared = model_->host_ref();
    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original incident energy/dir
    std::vector<real_type> const expected_energy = {9.9158593229731196e-06,
                                                    9.3982652856539062e-05,
                                                    0.00098086065952429635,
                                                    0.0098830010231267806,
                                                    0.093810968800312367,
                                                    0.93828180441505538,
                                                    9.9840299241598132,
                                                    99.943506172234038,
                                                    999.93677831821105,
                                                    9999.9543265397624};
    std::vector<real_type> const expected_angle = {0.73642517015232023,
                                                   -0.93575721388581834,
                                                   0.39720848171626949,
                                                   0.63289309823793904,
                                                   -0.99269103818369198,
                                                   -0.98604699538464768,
                                                   0.95040638380783515,
                                                   0.98326255459488243,
                                                   0.99871276520268837,
                                                   0.99997746197915383};

    real_type energy = 1e-5;
    real_type const factor = 1e+1;
    for (auto i : range(expected_energy.size()))
    {
        this->set_inc_particle(pdg::neutron(), MevEnergy{energy});
        ChipsNeutronElasticInteractor interact(
            shared, this->particle_track(), this->direction(), isotope_he4);
        Interaction result = interact(rng_engine);

        // Check scattered energy and angle at each incident neutron energy
        EXPECT_SOFT_EQ(result.energy.value(), expected_energy[i]);
        EXPECT_SOFT_EQ(dot_product(result.direction, this->direction()),
                       expected_angle[i]);
        energy *= factor;
    }
}

TEST_F(NeutronElasticTest, stress_test)
{
    // Set the target isotope : Cu65
    ElementComponentId el_id{1};
    IsotopeComponentId iso_id{1};
    IsotopeView const isotope = this->material_track()
                                    .make_material_view()
                                    .make_element_view(el_id)
                                    .make_isotope_view(iso_id);

    // Sample interaction
    NeutronElasticRef shared = model_->host_ref();
    RandomEngine& rng_engine = this->rng();

    // Produce samples from the incident energy
    std::vector<real_type> neutron_energy;
    real_type const expected_energy[] = {9.6937900401599116e-05,
                                         0.0096967590954386649,
                                         0.96066635424842617,
                                         99.915949160022208,
                                         9999.9519447032781};
    std::vector<real_type> cos_theta;
    real_type const expected_angle[] = {-0.0062993051693808555,
                                        0.0036712145032219813,
                                        -0.29680452456419526,
                                        0.97426025128623828,
                                        0.9999755334707946};

    int const num_sample = 100;
    real_type const weight = 1.0 / static_cast<real_type>(num_sample);
    std::vector<real_type> inc_energy = {1e-4, 0.01, 1., 100., 10000.};
    std::vector<int> avg_rng_samples;
    for (auto i : range(inc_energy.size()))
    {
        this->set_inc_particle(pdg::neutron(), MevEnergy{inc_energy[i]});
        ChipsNeutronElasticInteractor interact(
            shared, this->particle_track(), this->direction(), isotope);

        real_type sum_energy{0.};
        real_type sum_angle{0.};
        int sum_count{0};
        for ([[maybe_unused]] int j : range(num_sample))
        {
            // Sample scattered neutron energy and angle
            Interaction result = interact(rng_engine);
            sum_energy += result.energy.value();
            sum_angle += result.direction[2];
            sum_count += rng_engine.count();
            rng_engine.reset_count();
        }
        avg_rng_samples.push_back(sum_count / num_sample);
        EXPECT_SOFT_EQ(weight * sum_energy, expected_energy[i]);
        EXPECT_SOFT_EQ(weight * sum_angle, expected_angle[i]);
    }
    static int const expected_avg_rng_samples[] = {4, 4, 6, 6, 6};
    EXPECT_VEC_EQ(expected_avg_rng_samples, avg_rng_samples);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
