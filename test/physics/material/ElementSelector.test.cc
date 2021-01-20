//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ElementSelector.test.cc
//---------------------------------------------------------------------------//
#include "physics/material/ElementSelector.hh"

#include <memory>
#include <random>
#include "celeritas_test.hh"
#include "base/Range.hh"
#include "physics/material/MaterialParams.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ElementSelectorTest : public celeritas::Test
{
  public:
    //!@{
    //! Type aliases
    using RandomEngine = std::mt19937;
    //!@}

  protected:
    void SetUp() override
    {
        using celeritas::units::AmuMass;

        MaterialParams::Input inp;
        inp.elements = {
            {1, AmuMass{1.008}, "H"},
            {11, AmuMass{22.98976928}, "Na"},
            {13, AmuMass{26.9815385}, "Al"},
            {53, AmuMass{126.90447}, "I"},
        };
        inp.materials = {
            {0.0, 0.0, MatterState::unspecified, {}, "hard_vacuum"},
            {0.1 * constants::na_avogadro,
             293.0,
             MatterState::gas,
             {{ElementDefId{2}, 1.0}},
             "Al"},
            {0.05 * constants::na_avogadro,
             293.0,
             MatterState::solid,
             {{ElementDefId{0}, 0.25},
              {ElementDefId{1}, 0.25},
              {ElementDefId{2}, 0.25},
              {ElementDefId{3}, 0.25}},
             "everything_even"},
            {1 * constants::na_avogadro,
             293.0,
             MatterState::solid,
             {{ElementDefId{0}, 0.48},
              {ElementDefId{1}, 0.24},
              {ElementDefId{2}, 0.16},
              {ElementDefId{3}, 0.12}},
             "everything_weighted"},
        };
        mats      = std::make_shared<MaterialParams>(std::move(inp));
        host_mats = mats->host_pointers();

        // Allocate storage
        storage.assign(mats->max_element_components(), -1);
    }

    std::shared_ptr<MaterialParams> mats;
    MaterialParamsPointers          host_mats;
    RandomEngine                    rng;
    std::vector<real_type>          storage;
};

// Return cross section proportional to the element ID offset by 1.
real_type mock_micro_xs(ElementDefId el_id)
{
    CELER_EXPECT(el_id < 4);
    return static_cast<real_type>(el_id.get() + 1);
}

// Example functor for calculating cross section from actual atomic properties
// and particle state
struct CalcFancyMicroXs
{
    CalcFancyMicroXs(const MaterialParamsPointers& mats,
                     units::MevEnergy              energy)
        : mats_(mats), inv_energy_(1 / energy.value())
    {
    }

    real_type operator()(ElementDefId el_id) const
    {
        CELER_EXPECT(el_id);
        ElementView el(mats_, el_id);
        return el.cbrt_z() * inv_energy_;
    }

    const MaterialParamsPointers& mats_;
    real_type                     inv_energy_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

//! You can't select an element in pure void. (No interactions anyway.)
TEST_F(ElementSelectorTest, TEST_IF_CELERITAS_DEBUG(vacuum))
{
    MaterialView material(host_mats, mats->find("hard_vacuum"));
    EXPECT_THROW(ElementSelector(material, mock_micro_xs, make_span(storage)),
                 celeritas::DebugError);
}

//! Single element should always select the first one.
TEST_F(ElementSelectorTest, single)
{
    MaterialView    material(host_mats, mats->find("Al"));
    ElementSelector select_el(material, mock_micro_xs, make_span(storage));

    // Construction should have precalculated cross sections
    const double expected_elemental_micro_xs[] = {3};
    EXPECT_VEC_SOFT_EQ(expected_elemental_micro_xs,
                       select_el.elemental_micro_xs());
    EXPECT_SOFT_EQ(3.0, select_el.material_micro_xs());

    // Select a single element
    for (CELER_MAYBE_UNUSED auto i : range(100))
    {
        auto el_id = select_el(rng);
        EXPECT_EQ(ElementComponentId{0}, el_id);
    }
}

//! Equal number densities but unequal cross sections
TEST_F(ElementSelectorTest, everything_even)
{
    MaterialView    material(host_mats, mats->find("everything_even"));
    ElementSelector select_el(material, mock_micro_xs, make_span(storage));

    // Test cross sections
    const double expected_elemental_micro_xs[] = {1, 2, 3, 4};
    EXPECT_VEC_SOFT_EQ(expected_elemental_micro_xs,
                       select_el.elemental_micro_xs());
    EXPECT_SOFT_EQ(2.5, select_el.material_micro_xs());

    // Select a single element
    std::vector<int> tally(material.num_elements(), 0);
    for (CELER_MAYBE_UNUSED auto i : range(10000))
    {
        auto el_id = select_el(rng);
        ASSERT_LT(el_id.get(), tally.size());
        ++tally[el_id.get()];
    }

    // Proportional to micro_xs (equal number density)
    const int expected_tally[] = {1032, 2014, 2971, 3983};
    EXPECT_VEC_EQ(expected_tally, tally);
}

//! Number densities scaled to 1/xs so equiprobable
TEST_F(ElementSelectorTest, everything_weighted)
{
    MaterialView    material(host_mats, mats->find("everything_weighted"));
    ElementSelector select_el(material, mock_micro_xs, make_span(storage));

    // Test cross sections
    const double expected_elemental_micro_xs[] = {1, 2, 3, 4};
    EXPECT_VEC_SOFT_EQ(expected_elemental_micro_xs,
                       select_el.elemental_micro_xs());
    EXPECT_SOFT_EQ(1.92, select_el.material_micro_xs());

    // Select a single element
    std::vector<int> tally(material.num_elements(), 0);
    for (CELER_MAYBE_UNUSED auto i : range(10000))
    {
        auto el_id = select_el(rng);
        ASSERT_LT(el_id.get(), tally.size());
        ++tally[el_id.get()];
    }

    // Equiprobable
    const int expected_tally[] = {2574, 2395, 2589, 2442};
    EXPECT_VEC_EQ(expected_tally, tally);
}

//! Example of using a more complex/functional cross section functor
TEST_F(ElementSelectorTest, fancy_xs)
{
    units::MevEnergy energy{123};
    MaterialView     material(host_mats, mats->find("everything_weighted"));
    ElementSelector  select_el(
        material, CalcFancyMicroXs{host_mats, energy}, make_span(storage));

    // Test cross sections
    const double expected_elemental_micro_xs[] = {
        0.008130081300813, 0.01808113894772, 0.01911654217659, 0.0305389085709};
    EXPECT_VEC_SOFT_EQ(expected_elemental_micro_xs,
                       select_el.elemental_micro_xs());
    EXPECT_SOFT_EQ(0.014965228148605575, select_el.material_micro_xs());
}
