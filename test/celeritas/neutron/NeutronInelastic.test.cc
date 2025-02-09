//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/NeutronInelastic.test.cc
//---------------------------------------------------------------------------//
#include <memory>

#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/grid/TwodGridCalculator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridData.hh"
#include "celeritas/io/NeutronXsReader.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/neutron/NeutronTestBase.hh"
#include "celeritas/neutron/interactor/NeutronInelasticInteractor.hh"
#include "celeritas/neutron/interactor/detail/CascadeCollider.hh"
#include "celeritas/neutron/interactor/detail/CascadeParticle.hh"
#include "celeritas/neutron/model/CascadeOptions.hh"
#include "celeritas/neutron/model/NeutronInelasticModel.hh"
#include "celeritas/neutron/xs/NeutronInelasticMicroXsCalculator.hh"
#include "celeritas/neutron/xs/NucleonNucleonXsCalculator.hh"
#include "celeritas/phys/MacroXsCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class NeutronInelasticTest : public NeutronTestBase
{
  protected:
    using MevEnergy = units::MevEnergy;
    using SPConstNInelasticModel = std::shared_ptr<NeutronInelasticModel const>;

    void SetUp() override
    {
        using namespace units;

        // Load neutron elastic cross section data
        std::string data_path = this->test_data_path("celeritas", "");
        NeutronXsReader read_el_data(NeutronXsType::inel, data_path.c_str());

        // Set up the default particle: 100 MeV neutron along +z direction
        auto const& particles = *this->particle_params();
        this->set_inc_particle(pdg::neutron(), MevEnergy{100});
        this->set_inc_direction({0, 0, 1});

        // Build the model with the default material
        this->set_material("HeCu");
        CascadeOptions options;
        model_
            = std::make_shared<NeutronInelasticModel>(ActionId{0},
                                                      particles,
                                                      *this->material_params(),
                                                      options,
                                                      read_el_data);
    }

  protected:
    SPConstNInelasticModel model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(NeutronInelasticTest, micro_xs)
{
    // Calculate the elastic neutron-nucleus microscopic cross section
    using XsCalculator = NeutronInelasticMicroXsCalculator;

    // Set the target element: Cu
    ElementId el_id{1};

    // Check the size of the element cross section data (G4PARTICLEXS4.0)
    // The neutron/inelZ data are pruned by a factor of 5 for this test
    NeutronInelasticRef shared = model_->host_ref();
    NonuniformGridRecord grid = shared.micro_xs[el_id];
    EXPECT_EQ(grid.grid.size(), 61);

    // Microscopic cross section (units::BarnXs) in [1e-04:1e+4] (MeV)
    std::vector<real_type> const expected_micro_xs = {2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      0.2170446680979802,
                                                      1.3677671823188946,
                                                      0.81016638725225387,
                                                      0.84789596907525477};

    real_type energy = 1e-4;
    real_type const factor = 1e+1;
    for (auto i : range(expected_micro_xs.size()))
    {
        XsCalculator calc_micro_xs(shared, MevEnergy{energy});
        EXPECT_SOFT_EQ(calc_micro_xs(el_id).value(), expected_micro_xs[i]);
        energy *= factor;
    }

    // Check the elastic cross section at the upper bound (20 GeV)
    XsCalculator calc_upper_xs(shared, MevEnergy{2e+4});
    EXPECT_SOFT_EQ(calc_upper_xs(el_id).value(), 0.80300000000000027);
}

TEST_F(NeutronInelasticTest, macro_xs)
{
    // Calculate the inelastic neutron-nucleus macroscopic cross section
    auto material = this->material_track().make_material_view();
    auto calc_xs = MacroXsCalculator<NeutronInelasticMicroXsCalculator>(
        model_->host_ref(), material);

    // Macroscopic cross section (\f$ cm^{-1} \f$) in [1e-04:1e+4] (MeV)
    std::vector<real_type> const expected_macro_xs = {1.0577605656430734e-06,
                                                      4.4447010621996484e-07,
                                                      2.5134945234021254e-07,
                                                      1.9270371228950039e-07,
                                                      0.015057496086707027,
                                                      0.094888935102106969,
                                                      0.056850427191922973,
                                                      0.059657345679963072};

    real_type energy = 1e-4;
    real_type const factor = 1e+1;
    for (auto i : range(expected_macro_xs.size()))
    {
        EXPECT_SOFT_EQ(
            native_value_to<units::InvCmXs>(calc_xs(MevEnergy{energy})).value(),
            expected_macro_xs[i]);
        energy *= factor;
    }

    // Check the neutron inelastic interaction cross section at the upper bound
    // (20 GeV)
    EXPECT_SOFT_EQ(
        native_value_to<units::InvCmXs>(calc_xs(MevEnergy{2000})).value(),
        0.061219850473480573);
}

TEST_F(NeutronInelasticTest, nucleon_xs)
{
    // Calculate nucleon-nucleon cross sections (units::BarnXs)
    NeutronInelasticRef shared = model_->host_ref();

    NucleonNucleonXsCalculator calc_xs(shared);
    size_type num_channels = shared.scalars.num_channels();
    EXPECT_EQ(num_channels, 2);

    std::vector<real_type> xs_zero;
    std::vector<real_type> xs;
    for (auto ch_id : range(ChannelId{num_channels}))
    {
        xs_zero.push_back(shared.xs_params[ch_id].xs_zero);
        for (real_type inc_e : {0.01, 0.1, 1., 10., 100., 320.})
        {
            xs.push_back(calc_xs(ch_id, MevEnergy{inc_e}).value());
        }
    }
    real_type const expected_xs_zero[] = {17.613, 20.36};
    real_type const expected_xs[] = {17.613,
                                     17.613,
                                     4.0,
                                     0.8633,
                                     0.0691,
                                     0.0351,
                                     20.36,
                                     19.2,
                                     1.92,
                                     0.3024,
                                     0.0316,
                                     0.0233};
    EXPECT_VEC_SOFT_EQ(expected_xs_zero, xs_zero);
    EXPECT_VEC_SOFT_EQ(expected_xs, xs);
}

TEST_F(NeutronInelasticTest, model_data)
{
    // Test neutron inelastic interactions
    NeutronInelasticRef shared = model_->host_ref();

    // Set the target to (light) isotope He3
    IsotopeId iso_id{0};

    // Check the size of the number of nuclear zones
    NuclearZones he3_nuclear_zones = shared.nuclear_zones.zones[iso_id];
    EXPECT_EQ(he3_nuclear_zones.zones.size(), 1);

    // Check zone data
    for (auto sid : he3_nuclear_zones.zones)
    {
        ZoneComponent components = shared.nuclear_zones.components[sid];
        EXPECT_SOFT_EQ(8, components.radius);
        EXPECT_SOFT_EQ(2144.6605848506319, components.volume);
        // proton
        EXPECT_SOFT_EQ(0.00093254849467907434, components.density[0]);
        EXPECT_SOFT_EQ(188.75462299392046, components.fermi_mom[0]);
        EXPECT_SOFT_EQ(24.476129399543886, components.potential[0]);
        // neutron
        EXPECT_SOFT_EQ(0.00046627424733953717, components.density[1]);
        EXPECT_SOFT_EQ(149.81464355220513, components.fermi_mom[1]);
        EXPECT_SOFT_EQ(55.944047271729367, components.potential[1]);
    }

    // Set the target to (heavy) isotope Cu63
    IsotopeId iso_cu63{2};

    // Check the size of the number of nuclear zones
    NuclearZones cu63_nuclear_zones = shared.nuclear_zones.zones[iso_cu63];
    EXPECT_EQ(cu63_nuclear_zones.zones.size(), 3);

    // Check zone data
    std::vector<real_type> radii;
    std::vector<real_type> volumes;
    std::vector<real_type> densities;
    std::vector<real_type> fermi_moms;
    std::vector<real_type> potentials;

    for (auto sid : cu63_nuclear_zones.zones)
    {
        ZoneComponent components = shared.nuclear_zones.components[sid];
        radii.push_back(components.radius);
        volumes.push_back(components.volume);
        for (auto nucleon_index : range(2))
        {
            densities.push_back(components.density[nucleon_index]);
            fermi_moms.push_back(components.fermi_mom[nucleon_index]);
            potentials.push_back(components.potential[nucleon_index]);
        }
    }

    real_type const expected_cu_radii[]
        = {12.0056427171327, 14.924785000235, 21.383497297248};

    real_type const expected_cu_volumes[]
        = {7248.44509638664, 6677.12103595803, 27031.1207141316};

    // clang-format off
    real_type const expected_cu_densities[]
        = {0.00218857360950474, 0.00256591388700555, 0.0011959208904212,
           0.00140211414739038, 0.000190555762441557, 0.000223410204241825};

    real_type const expected_cu_fermi_moms[]
        = {250.838488281962, 264.497241970299, 205.072742538403,
           216.239442265022, 111.176880545669, 117.23072673814};

    real_type const expected_cu_potentials[]
        = {39.6516941248423, 48.0933349774238, 28.5327876764636,
           35.7475768798981, 12.7087352945686, 18.1775106385426};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_cu_radii, radii);
    EXPECT_VEC_SOFT_EQ(expected_cu_volumes, volumes);
    EXPECT_VEC_SOFT_EQ(expected_cu_densities, densities);
    EXPECT_VEC_SOFT_EQ(expected_cu_fermi_moms, fermi_moms);
    EXPECT_VEC_SOFT_EQ(expected_cu_potentials, potentials);

    // Set the target to (very heavy, A > 100) isotope Pb208
    IsotopeId iso_pb208{7};

    // Check the size of the number of nuclear zones
    NuclearZones pb208_nuclear_zones = shared.nuclear_zones.zones[iso_pb208];
    EXPECT_EQ(pb208_nuclear_zones.zones.size(), 6);

    // Check zone data
    radii.clear();
    volumes.clear();
    densities.clear();
    fermi_moms.clear();
    potentials.clear();

    for (auto sid : pb208_nuclear_zones.zones)
    {
        ZoneComponent components = shared.nuclear_zones.components[sid];
        radii.push_back(components.radius);
        volumes.push_back(components.volume);
        for (auto nucleon_index : range(2))
        {
            densities.push_back(components.density[nucleon_index]);
            fermi_moms.push_back(components.fermi_mom[nucleon_index]);
            potentials.push_back(components.potential[nucleon_index]);
        }
    }

    // clang-format off
    real_type const expected_pb_radii[]
        = {16.2612786504728, 19.3490859199713, 20.7466319700195,
           22.4369887370502, 23.834545403954, 25.122295335588};

    real_type const expected_pb_volumes[]
        = {18011.6162284334, 12332.183871369, 7061.35140601041,
           9908.04814765151, 9403.27500414658, 9698.58388620079};

    real_type const expected_pb_densities[]
        = {0.00225402783765291,  0.00346350618956423,  0.00177111372940623,
           0.00272146743786811,  0.00115221167124763,  0.0017704715924049,
           0.000673003348019808, 0.00103412709573775,  0.000333946126232526,
           0.000513136730552418, 0.000166530199539091, 0.000255887867584456};

    real_type const expected_pb_fermi_moms[]
        = {253.314595582345, 292.311416929582, 233.752324669988,
           269.737608596091, 202.5432986892,   233.72407141927,
           169.308799502358, 195.373247117507, 134.039436095239,
           154.674298965551, 106.293030558995, 122.656439519448};

    real_type const expected_pb_potentials[]
        = {42.1989261226909, 52.8390035394296, 37.1214353127717,
           46.0871655105693, 29.8653511196685, 36.4383237834904,
           23.279671229538, 27.6809580702491, 17.5782866566401,
           20.0994918268739, 14.0247531445444, 15.3741494083458};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_pb_radii, radii);
    EXPECT_VEC_SOFT_EQ(expected_pb_volumes, volumes);
    EXPECT_VEC_SOFT_EQ(expected_pb_densities, densities);
    EXPECT_VEC_SOFT_EQ(expected_pb_fermi_moms, fermi_moms);
    EXPECT_VEC_SOFT_EQ(expected_pb_potentials, potentials);

    // Set the target to isotope B11 and validate the Gaussian potential
    IsotopeId iso_b11{9};

    // Check the size of the number of nuclear zones
    NuclearZones b11_nuclear_zones = shared.nuclear_zones.zones[iso_b11];
    EXPECT_EQ(b11_nuclear_zones.zones.size(), 3);

    // Clear zone data
    radii.clear();
    volumes.clear();
    densities.clear();
    fermi_moms.clear();
    potentials.clear();

    for (auto sid : b11_nuclear_zones.zones)
    {
        ZoneComponent components = shared.nuclear_zones.components[sid];
        radii.push_back(components.radius);
        volumes.push_back(components.volume);
        for (auto nucleon_index : range(2))
        {
            densities.push_back(components.density[nucleon_index]);
            fermi_moms.push_back(components.fermi_mom[nucleon_index]);
            potentials.push_back(components.potential[nucleon_index]);
        }
    }

    real_type const expected_b_radii[]
        = {4.54355900675881, 8.34774540792328, 16.3261316488719};

    real_type const expected_b_volumes[]
        = {392.895565424687, 2043.77151146382, 15791.3107610447};

    // clang-format off
    real_type const expected_b_densities[]
        = {0.00169984536247725, 0.0020398144349727, 0.000949798766722706,
           0.00113975852006725, 0.00015141027051547, 0.000181692324618564};

    real_type const expected_b_fermi_moms[]
        = {230.573961004183, 245.021395491472, 189.911379816935,
	   201.810955147759, 102.973522012059, 109.425695565028};

    real_type const expected_b_potentials[]
        = {39.5599907769562, 43.4025388663509, 30.4485502393036,
	   33.1276701038221, 16.8795715233179, 17.8260857964719};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_b_radii, radii);
    EXPECT_VEC_SOFT_EQ(expected_b_volumes, volumes);
    EXPECT_VEC_SOFT_EQ(expected_b_densities, densities);
    EXPECT_VEC_SOFT_EQ(expected_b_fermi_moms, fermi_moms);
    EXPECT_VEC_SOFT_EQ(expected_b_potentials, potentials);
}

TEST_F(NeutronInelasticTest, angular_cdf)
{
    // Check the tabulated cummulative distribution function (c.d.f) data used
    // for sampling cos\theta of the intra-nucleus nucleon-nucleon collision
    NeutronInelasticRef shared = model_->host_ref();

    TwodGridData grid_cdf = shared.angular_cdf[ChannelId{0}];
    NonuniformGrid<real_type> const y_grid{grid_cdf.y, shared.reals};
    TwodGridCalculator calc_cdf(grid_cdf, shared.reals);

    EXPECT_EQ(shared.angular_cdf.size(), 2);
    EXPECT_EQ(grid_cdf.x.size(), 6);
    EXPECT_EQ(grid_cdf.y.size(), 19);
    EXPECT_EQ(grid_cdf.values.size(), 114);
    EXPECT_EQ(calc_cdf({0, -1}), 0);

    // Calculate the c.d.f value at a given point of energy and cos\theta
    std::vector<real_type> cdf;
    for (real_type inc_e : {50., 150., 250.})
    {
        for (real_type cos_theta : {-0.5, 0., 0.5})
        {
            cdf.push_back(calc_cdf({inc_e, cos_theta}));
        }
    }

    real_type const expected_cdf[] = {0.25583333333333,
                                      0.5,
                                      0.74416666666667,
                                      0.264,
                                      0.5,
                                      0.736,
                                      0.25795,
                                      0.5,
                                      0.74205};

    EXPECT_VEC_SOFT_EQ(expected_cdf, cdf);
}

TEST_F(NeutronInelasticTest, cascade_collider)
{
    using ParticleType = CascadeParticle::ParticleType;
    using FinalState = Array<CascadeParticle, 2>;

    // Test nuleon-nucleon intra-nuclear cascade interactions
    NeutronInelasticRef shared = model_->host_ref();
    RandomEngine& rng_engine = this->rng();

    // Neutron-neutron channel
    CascadeParticle bullet{ParticleType::neutron,
                           shared.scalars.neutron_mass,
                           {{23.1269, -25.7938, 603.536}, 1117.25}};
    CascadeParticle target{ParticleType::neutron,
                           shared.scalars.neutron_mass,
                           {{29.5146, -127.849, 83.849}, 952.381}};

    detail::CascadeCollider nn_collider(shared, bullet, target);
    FinalState nn_result = nn_collider(rng_engine);

    // Neutron-proton channel
    bullet = {ParticleType::neutron,
              shared.scalars.neutron_mass,
              {{289.968, -55.9082, 22.8334}, 985.145}};
    target = {ParticleType::proton,
              shared.scalars.proton_mass,
              {{-43.6161, 112.144, -146.784}, 957.277}};

    detail::CascadeCollider np_collider(shared, bullet, target);
    FinalState np_result = np_collider(rng_engine);

    // Expected results (verified with Geant4 11.2.2)
    real_type expected_nn_energy[] = {985.07411365134317, 1084.5568863486567};
    real_type expected_np_energy[] = {985.94825543412617, 956.47374456587363};
    Real3 expected_np_mom[]
        = {{295.582760097549, -7.33588399587099, 43.4489730961597},
           {-49.2308600975493, 63.571683995871, -167.39957309616}};
    Real3 expected_nn_mom[]
        = {{-91.1374627277095, -224.52912872407, 169.907259286778},
           {143.77896272771, 70.8863287240703, 517.477740713221}};

    for (auto i : range(2))
    {
        EXPECT_SOFT_EQ(expected_nn_energy[i], nn_result[i].four_vec.energy);
        EXPECT_SOFT_EQ(expected_np_energy[i], np_result[i].four_vec.energy);
        EXPECT_VEC_SOFT_EQ(expected_nn_mom[i], nn_result[i].four_vec.mom);
        EXPECT_VEC_SOFT_EQ(expected_np_mom[i], np_result[i].four_vec.mom);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
