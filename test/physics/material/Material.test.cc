//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Material.test.cc
//---------------------------------------------------------------------------//
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/MaterialParams.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/material/MaterialStatePointers.hh"
#include "physics/material/detail/Utils.hh"

#include <limits>
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "base/DeviceVector.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementDef.hh"
#include "Material.test.hh"

using namespace celeritas;
using celeritas::units::AmuMass;
using celeritas_test::m_test;
using celeritas_test::MTestInput;

//---------------------------------------------------------------------------//
/*!
 * Test mass radiation coefficient calculation.
 *
 * Reference values are from
 * https://pdg.lbl.gov/2020/AtomicNuclearProperties/index.html
 *
 * We test all the special cases (H through Li) plus the regular case plus the
 * "out-of-bounds" case for transuranics.
 *
 * The mass radiation coefficient (the inverse of which is referred to as
 * "radiation length" in the PDG physics review) is an inverse length, divided
 * by the material density: units are [cm^2/g]. It's analogous to the mass
 * attenutation coefficient mu / rho.
 */
TEST(MaterialUtils, radiation_length)
{
    using celeritas::detail::calc_mass_rad_coeff;
    using celeritas::units::AmuMass;

    ElementDef el;

    // Hydrogen
    el.atomic_number = 1;
    el.atomic_mass   = AmuMass{1.008};
    EXPECT_SOFT_NEAR(63.04, 1 / calc_mass_rad_coeff(el), 1e-3);
    real_type hydrogen_density = 8.376e-05; // g/cc
    EXPECT_SOFT_NEAR(
        7.527e5, 1 / (hydrogen_density * calc_mass_rad_coeff(el)), 1e-3);

    // Helium
    el.atomic_number = 2;
    el.atomic_mass   = AmuMass{4.002602};
    EXPECT_SOFT_NEAR(94.32, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Lithium
    el.atomic_number = 3;
    el.atomic_mass   = AmuMass{6.94};
    EXPECT_SOFT_NEAR(82.77, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Beryllium
    el.atomic_number = 4;
    el.atomic_mass   = AmuMass{9.0121831};
    EXPECT_SOFT_NEAR(65.19, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Aluminum
    el.atomic_number = 13;
    el.atomic_mass   = AmuMass{26.9815385};
    EXPECT_SOFT_NEAR(24.01, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Uranium
    el.atomic_number = 92;
    el.atomic_mass   = AmuMass{238.02891};
    EXPECT_SOFT_NEAR(6.00, 1 / calc_mass_rad_coeff(el), 1e-3);

    // Plutonium-244 [NOTE: accuracy decreases compared to tabulated values]
    el.atomic_number = 94;
    el.atomic_mass   = AmuMass{244.06420};
    EXPECT_SOFT_NEAR(5.93, 1 / calc_mass_rad_coeff(el), 1e-2);
}

//---------------------------------------------------------------------------//
// MATERIALS HOST TEST
//---------------------------------------------------------------------------//

class MaterialTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        MaterialParams::Input inp;
        inp.elements = {
            {1, AmuMass{1.008}, "H"},
            {13, AmuMass{26.9815385}, "Al"},
            {11, AmuMass{22.98976928}, "Na"},
            {53, AmuMass{126.90447}, "I"},
        };
        inp.materials = {
            {0.05 * constants::na_avogadro, // XXX fix number density
             293.0,
             MatterState::solid,
             {{ElementDefId{2}, 0.5}, {ElementDefId{3}, 0.5}},
             "NaI"},
            {0.0, 0.0, MatterState::unspecified, {}, "hard vacuum"},
            {1e-5 * constants::na_avogadro, // XXX fix number density
             100.0,
             MatterState::gas,
             {{ElementDefId{0}, 1.0}},
             "H2"},
        };
        params = std::make_shared<MaterialParams>(std::move(inp));
    }

    std::shared_ptr<MaterialParams> params;
};

//---------------------------------------------------------------------------//

TEST_F(MaterialTest, params)
{
    ASSERT_TRUE(params);
    EXPECT_EQ(MaterialDefId{0}, params->find("NaI"));
    EXPECT_EQ(MaterialDefId{1}, params->find("hard vacuum"));
    EXPECT_EQ(MaterialDefId{2}, params->find("H2"));
    EXPECT_EQ(MaterialDefId{}, params->find("nonexistent material"));

    EXPECT_EQ("H", params->id_to_label(ElementDefId{0}));
    EXPECT_EQ("Al", params->id_to_label(ElementDefId{1}));
    EXPECT_EQ("Na", params->id_to_label(ElementDefId{2}));
    EXPECT_EQ("I", params->id_to_label(ElementDefId{3}));

    EXPECT_EQ("NaI", params->id_to_label(MaterialDefId{0}));
    EXPECT_EQ("hard vacuum", params->id_to_label(MaterialDefId{1}));
    EXPECT_EQ("H2", params->id_to_label(MaterialDefId{2}));
}

TEST_F(MaterialTest, material_view)
{
    auto host_ptrs = params->host_pointers();
    {
        // NaI
        // TODO: update density and check against geant4 values
        MaterialView mat(host_ptrs, MaterialDefId{0});
        EXPECT_SOFT_EQ(0.05, mat.number_density() / constants::na_avogadro);
        EXPECT_SOFT_EQ(293.0, mat.temperature());
        EXPECT_EQ(MatterState::solid, mat.matter_state());
        EXPECT_SOFT_EQ(3.7473559807049948, mat.density());
        EXPECT_SOFT_EQ(9.635425216e+23, mat.electron_density());
        EXPECT_SOFT_EQ(3.4662701145954062, mat.radiation_length());

        // Test element view
        auto els = mat.elements();
        ASSERT_EQ(2, els.size());
        EXPECT_EQ(ElementDefId{2}, els[0].element);
        EXPECT_SOFT_EQ(0.5, els[0].fraction);
        EXPECT_EQ(ElementDefId{3}, els[1].element);
        EXPECT_SOFT_EQ(0.5, els[1].fraction);
    }
    {
        // vacuum
        MaterialView mat(host_ptrs, MaterialDefId{1});
        EXPECT_SOFT_EQ(0.0, mat.number_density());
        EXPECT_SOFT_EQ(0.0, mat.temperature());
        EXPECT_EQ(MatterState::unspecified, mat.matter_state());
        EXPECT_SOFT_EQ(0.0, mat.density());
        EXPECT_SOFT_EQ(0.0, mat.electron_density()); // FIXME
        EXPECT_SOFT_EQ(std::numeric_limits<real_type>::infinity(),
                       mat.radiation_length());

        // Test element view
        auto els = mat.elements();
        ASSERT_EQ(0, els.size());
    }
}

TEST_F(MaterialTest, element_view)
{
    auto host_ptrs = params->host_pointers();
    {
        // Test aluminum
        ElementView el(host_ptrs, ElementDefId{1});
        EXPECT_EQ(13, el.atomic_number());
        EXPECT_SOFT_EQ(26.9815385, el.atomic_mass().value());
        EXPECT_SOFT_EQ(std::pow(13.0, 1.0 / 3), el.cbrt_z());
        EXPECT_SOFT_EQ(std::pow(13.0 * 14.0, 1.0 / 3), el.cbrt_zzp());
        EXPECT_SOFT_EQ(std::log(13.0), el.log_z());
        EXPECT_SOFT_EQ(0.041647232662906583, el.mass_radiation_coeff());
    }
}

#if CELERITAS_USE_CUDA
class MaterialDeviceTest : public MaterialTest
{
    using Base = MaterialTest;

  protected:
    void SetUp() override { Base::SetUp(); }
};

TEST_F(MaterialDeviceTest, all)
{
    MTestInput input;
    input.init = {{MaterialDefId{0}}, {MaterialDefId{1}}, {MaterialDefId{2}}};

    DeviceVector<MaterialTrackState> states(input.init.size());
    input.params       = params->device_pointers();
    input.states.state = states.device_pointers();

    // Run GPU test
    auto result = m_test(input);

    const double expected_temperatures[] = {293, 0, 100};
    const double expected_rad_len[] = {3.466270114595, inf, 6254684.974443};
    const double expected_tot_z[]   = {9.635425216e+23, 0, 6.02214076e+18};

    EXPECT_VEC_SOFT_EQ(expected_temperatures, result.temperatures);
    EXPECT_VEC_SOFT_EQ(expected_rad_len, result.rad_len);
    EXPECT_VEC_SOFT_EQ(expected_tot_z, result.tot_z);
}

#endif
