//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/Material.test.cc
//---------------------------------------------------------------------------//
#include "Material.test.hh"

#include <cstring>
#include <limits>

#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialParamsOutput.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/mat/detail/Utils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, MaterialId const& mat)
{
    os << "MaterialId{";
    if (mat)
        os << mat.unchecked_get();
    os << "}";
    return os;
}

namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
//! Test coulomb correction values
TEST(MaterialUtils, coulomb_correction)
{
    constexpr AtomicNumber h{1};
    constexpr AtomicNumber al{13};
    constexpr AtomicNumber u{92};
    EXPECT_SOFT_EQ(6.4008218033384263e-05, calc_coulomb_correction(h));
    EXPECT_SOFT_EQ(0.010734632775699565, calc_coulomb_correction(al));
    EXPECT_SOFT_EQ(0.39494589680653375, calc_coulomb_correction(u));
}

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
    auto calc_inv_rad_coeff
        = [](int atomic_number, real_type amu_mass) -> real_type {
        ElementRecord el;
        el.atomic_number = AtomicNumber{atomic_number};
        el.atomic_mass = units::AmuMass{amu_mass};
        el.coulomb_correction = calc_coulomb_correction(el.atomic_number);

        return 1 / detail::calc_mass_rad_coeff(el);
    };

    // Hydrogen
    EXPECT_SOFT_NEAR(63.04, calc_inv_rad_coeff(1, 1.008), 1e-3);
    // Helium
    EXPECT_SOFT_NEAR(94.32, calc_inv_rad_coeff(2, 4.002602), 1e-3);
    // Lithium
    EXPECT_SOFT_NEAR(82.77, calc_inv_rad_coeff(3, 6.94), 1e-3);
    // Beryllium
    EXPECT_SOFT_NEAR(65.19, calc_inv_rad_coeff(4, 9.0121831), 1e-3);
    // Aluminum
    EXPECT_SOFT_NEAR(24.01, calc_inv_rad_coeff(13, 26.9815385), 1e-3);
    // Uranium
    EXPECT_SOFT_NEAR(6.00, calc_inv_rad_coeff(92, 238.02891), 1e-3);
    // Plutonium-244 [NOTE: accuracy decreases compared to tabulated values]
    EXPECT_SOFT_NEAR(5.93, calc_inv_rad_coeff(94, 244.06420), 1e-2);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail

namespace test
{
//---------------------------------------------------------------------------//
class MaterialTest : public Test
{
  protected:
    void SetUp() override
    {
        MaterialParams::Input inp;
        inp.elements = {
            {AtomicNumber{1}, units::AmuMass{1.008}, "H"},
            {AtomicNumber{13}, units::AmuMass{26.9815385}, "Al"},
            {AtomicNumber{11}, units::AmuMass{22.98976928}, "Na"},
            {AtomicNumber{53}, units::AmuMass{126.90447}, "I"},
        };
        inp.materials = {
            // Sodium iodide
            {2.948915064677e+22,
             293.0,
             MatterState::solid,
             {{ElementId{2}, 0.5}, {ElementId{3}, 0.5}},
             "NaI"},
            // Void
            {0, 0, MatterState::unspecified, {}, "hard vacuum"},
            // Diatomic hydrogen
            {1.0739484359044669e+20,
             100.0,
             MatterState::gas,
             {{ElementId{0}, 1.0}},
             Label{"H2", "1"}},
            // Diatomic hydrogen with the same name and different properties
            {1.072e+20,
             110.0,
             MatterState::gas,
             {{ElementId{0}, 1.0}},
             Label{"H2", "2"}},
        };
        params = std::make_shared<MaterialParams>(std::move(inp));
    }

    std::shared_ptr<MaterialParams> params;
};

//---------------------------------------------------------------------------//

TEST_F(MaterialTest, params)
{
    ASSERT_TRUE(params);

    EXPECT_EQ(4, params->size());
    EXPECT_EQ(4, params->num_materials());
    EXPECT_EQ(4, params->num_elements());

    EXPECT_EQ(MaterialId{0}, params->find_material("NaI"));
    EXPECT_EQ(MaterialId{1}, params->find_material("hard vacuum"));
    EXPECT_THROW(params->find_material("H2"), RuntimeError);
    {
        auto found = params->find_materials("H2");
        const MaterialId expected[] = {MaterialId{2}, MaterialId{3}};
        EXPECT_VEC_EQ(expected, found);
    }
    EXPECT_EQ(MaterialId{}, params->find_material("nonexistent material"));

    EXPECT_EQ("H", params->id_to_label(ElementId{0}).name);
    EXPECT_EQ("Al", params->id_to_label(ElementId{1}).name);
    EXPECT_EQ("Na", params->id_to_label(ElementId{2}).name);
    EXPECT_EQ("I", params->id_to_label(ElementId{3}).name);
    EXPECT_EQ(ElementId{1}, params->find_element("Al"));

    EXPECT_EQ("NaI", params->id_to_label(MaterialId{0}).name);
    EXPECT_EQ("hard vacuum", params->id_to_label(MaterialId{1}).name);
    EXPECT_EQ(Label("H2", "1"), params->id_to_label(MaterialId{2}));
    EXPECT_EQ(Label("H2", "2"), params->id_to_label(MaterialId{3}));

    EXPECT_EQ(2, params->max_element_components());
}

TEST_F(MaterialTest, material_view)
{
    {
        // NaI
        MaterialView mat = params->get(MaterialId{0});
        EXPECT_SOFT_EQ(2.948915064677e+22, mat.number_density());
        EXPECT_SOFT_EQ(293.0, mat.temperature());
        EXPECT_EQ(MatterState::solid, mat.matter_state());
        EXPECT_SOFT_EQ(32.0, mat.zeff());
        EXPECT_SOFT_EQ(3.6700020622594716, mat.density());
        EXPECT_SOFT_EQ(9.4365282069663997e+23, mat.electron_density());
        EXPECT_SOFT_EQ(3.5393292693170424, mat.radiation_length());
        EXPECT_SOFT_EQ(400.00760709482647e-6,
                       mat.mean_excitation_energy().value());
        EXPECT_SOFT_EQ(std::log(400.00760709482647e-6),
                       mat.log_mean_excitation_energy().value());

        // Test element view
        auto els = mat.elements();
        ASSERT_EQ(2, els.size());
        EXPECT_EQ(ElementId{2}, els[0].element);
        EXPECT_SOFT_EQ(0.5, els[0].fraction);
        EXPECT_EQ(ElementId{3}, els[1].element);
        EXPECT_SOFT_EQ(0.5, els[1].fraction);
    }
    {
        // vacuum
        MaterialView mat = params->get(MaterialId{1});
        EXPECT_SOFT_EQ(0, mat.number_density());
        EXPECT_SOFT_EQ(0, mat.temperature());
        EXPECT_EQ(MatterState::unspecified, mat.matter_state());
        EXPECT_SOFT_EQ(0, mat.zeff());
        EXPECT_SOFT_EQ(0, mat.density());
        EXPECT_SOFT_EQ(0, mat.electron_density());
        EXPECT_SOFT_EQ(std::numeric_limits<real_type>::infinity(),
                       mat.radiation_length());
        EXPECT_SOFT_EQ(0, mat.mean_excitation_energy().value());
        EXPECT_SOFT_EQ(-std::numeric_limits<real_type>::infinity(),
                       mat.log_mean_excitation_energy().value());

        // Test element view
        auto els = mat.elements();
        ASSERT_EQ(0, els.size());
    }
    {
        // H2
        MaterialView mat = params->get(MaterialId{2});
        EXPECT_SOFT_EQ(1.0739484359044669e+20, mat.number_density());
        EXPECT_SOFT_EQ(100, mat.temperature());
        EXPECT_EQ(MatterState::gas, mat.matter_state());
        EXPECT_SOFT_EQ(1.0, mat.zeff());
        EXPECT_SOFT_EQ(0.00017976, mat.density());
        EXPECT_SOFT_EQ(1.0739484359044669e+20, mat.electron_density());
        EXPECT_SOFT_EQ(350729.99844063615, mat.radiation_length());
        EXPECT_SOFT_EQ(19.2e-6, mat.mean_excitation_energy().value());
        EXPECT_SOFT_EQ(std::log(19.2e-6),
                       mat.log_mean_excitation_energy().value());

        // Test element view
        auto els = mat.elements();
        ASSERT_EQ(1, els.size());
    }
    {
        // H2_3
        MaterialView mat = params->get(MaterialId{3});
        EXPECT_SOFT_EQ(1.072e+20, mat.number_density());
        EXPECT_SOFT_EQ(110, mat.temperature());
        EXPECT_EQ(MatterState::gas, mat.matter_state());
        EXPECT_SOFT_EQ(1.0, mat.zeff());
        EXPECT_SOFT_EQ(0.00017943386624303615, mat.density());
        EXPECT_SOFT_EQ(1.072e+20, mat.electron_density());
        EXPECT_SOFT_EQ(351367.47504673258, mat.radiation_length());
        EXPECT_SOFT_EQ(19.2e-6, mat.mean_excitation_energy().value());
        EXPECT_SOFT_EQ(std::log(19.2e-6),
                       mat.log_mean_excitation_energy().value());

        // Test element view
        auto els = mat.elements();
        ASSERT_EQ(1, els.size());
    }
}

TEST_F(MaterialTest, element_view)
{
    {
        // Test aluminum
        ElementView el = params->get(ElementId{1});
        EXPECT_EQ(AtomicNumber{13}, el.atomic_number());
        EXPECT_SOFT_EQ(26.9815385, el.atomic_mass().value());
        EXPECT_SOFT_EQ(std::pow(13.0, 1.0 / 3), el.cbrt_z());
        EXPECT_SOFT_EQ(std::pow(13.0 * 14.0, 1.0 / 3), el.cbrt_zzp());
        EXPECT_SOFT_EQ(std::log(13.0), el.log_z());
        EXPECT_SOFT_EQ(0.010734632775699565, el.coulomb_correction());
        EXPECT_SOFT_EQ(0.04164723292591279, el.mass_radiation_coeff());
    }
}

TEST_F(MaterialTest, output)
{
    MaterialParamsOutput out(params);
    EXPECT_EQ("material", out.label());

    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ(
            R"json({"_units":{"atomic_mass":"amu","mean_excitation_energy":"MeV"},"elements":{"atomic_mass":[1.008,26.9815385,22.98976928,126.90447],"atomic_number":[1,13,11,53],"coulomb_correction":[6.400821803338426e-05,0.010734632775699565,0.00770256745342534,0.15954439947436763],"label":["H","Al","Na","I"],"mass_radiation_coeff":[0.0158611264432063,0.04164723292591279,0.03605392839455309,0.11791841505608874]},"materials":{"density":[3.6700020622594716,0.0,0.00017976000000000003,0.00017943386624303615],"electron_density":[9.4365282069664e+23,0.0,1.073948435904467e+20,1.072e+20],"element_frac":[[0.5,0.5],[],[1.0],[1.0]],"element_id":[[2,3],[],[0],[0]],"label":["NaI","hard vacuum","H2@1","H2@2"],"matter_state":["solid","unspecified","gas","gas"],"mean_excitation_energy":[0.00040000760709482647,0.0,1.9199999999999986e-05,1.9199999999999986e-05],"number_density":[2.948915064677e+22,0.0,1.073948435904467e+20,1.072e+20],"radiation_length":[3.5393292693170424,null,350729.99844063615,351367.4750467326],"temperature":[293.0,0.0,100.0,110.0],"zeff":[32.0,0.0,1.0,1.0]}})json",
            to_string(out))
            << "\n/*** REPLACE ***/\nR\"json(" << to_string(out)
            << ")json\"\n/******/";
    }
}

//---------------------------------------------------------------------------//
// IMPORT MATERIAL DATA TEST
//---------------------------------------------------------------------------//

class MaterialParamsImportTest : public Test
{
  protected:
    void SetUp() override
    {
        root_filename_
            = this->test_data_path("celeritas", "four-steel-slabs.root");
        RootImporter import_from_root(root_filename_.c_str());
        data_ = import_from_root();
    }
    std::string root_filename_;
    ImportData data_;

    ScopedRootErrorHandler scoped_root_error_;
};

TEST_F(MaterialParamsImportTest, TEST_IF_CELERITAS_USE_ROOT(import_materials))
{
    auto const material_params = MaterialParams::from_import(data_);
    // Material labels
    EXPECT_EQ("G4_Galactic", material_params->id_to_label(MaterialId{0}).name);
    EXPECT_EQ("G4_STAINLESS-STEEL",
              material_params->id_to_label(MaterialId{1}).name);

    /*!
     * Material
     *
     * Geant4 has outdated constants. The discrepancy between Geant4 /
     * Celeritas constants results in the slightly different numerical values
     * calculated by Celeritas.
     */
    MaterialView mat(material_params->host_ref(), MaterialId{1});

    EXPECT_EQ(MatterState::solid, mat.matter_state());
    EXPECT_SOFT_EQ(293.15, mat.temperature());  // [K]
    EXPECT_SOFT_EQ(7.9999999972353661, mat.density());  // [g/cm^3]
    EXPECT_SOFT_EQ(2.2444320228819809e+24,
                   mat.electron_density());  // [1/cm^3]
    EXPECT_SOFT_EQ(8.6993489258991514e+22, mat.number_density());  // [1/cm^3]

    // Test elements by unpacking them
    std::vector<unsigned int> els;
    std::vector<real_type> fracs;
    for (auto const& component : mat.elements())
    {
        els.push_back(component.element.unchecked_get());
        fracs.push_back(component.fraction);
    }

    // Fractions are normalized and thus may differ from the imported ones
    // Fe, Cr, Ni
    static unsigned int const expected_els[] = {0, 1, 2};
    static real_type expected_fracs[] = {0.74, 0.18, 0.08};
    EXPECT_VEC_EQ(expected_els, els);
    EXPECT_VEC_SOFT_EQ(expected_fracs, fracs);
}

//---------------------------------------------------------------------------//
// MATERIALS DEVICE TEST
//---------------------------------------------------------------------------//

class MaterialDeviceTest : public MaterialTest
{
    using Base = MaterialTest;

  protected:
    void SetUp() override { Base::SetUp(); }
};

TEST_F(MaterialDeviceTest, TEST_IF_CELER_DEVICE(all))
{
    MTestInput input;
    input.init
        = {{MaterialId{0}}, {MaterialId{1}}, {MaterialId{2}}, {MaterialId{3}}};

    CollectionStateStore<MaterialStateData, MemSpace::device> states(
        params->host_ref(), input.init.size());

    input.params = params->device_ref();
    input.states = states.ref();

    EXPECT_EQ(params->max_element_components(),
              input.params.max_element_components);
    EXPECT_EQ(4, input.states.state.size());
    EXPECT_EQ(4 * params->max_element_components(),
              input.states.element_scratch.size());

    // Run GPU test
    MTestOutput result;
#if CELER_USE_DEVICE
    result = m_test(input);
#endif

    double const expected_temperatures[] = {293, 0, 100, 110};
    double const expected_rad_len[]
        = {3.5393292693170424, inf, 350729.99844063615, 351367.47504673258};
    double const expected_tot_z[]
        = {9.4365282069664e+23, 0, 1.07394843590447e+20, 1.072e20};

    EXPECT_VEC_SOFT_EQ(expected_temperatures, result.temperatures);
    EXPECT_VEC_SOFT_EQ(expected_rad_len, result.rad_len);
    EXPECT_VEC_SOFT_EQ(expected_tot_z, result.tot_z);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
