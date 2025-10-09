//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/ElementSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/ElementSelector.hh"

#include <memory>
#include <random>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/random/SequenceEngine.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using MaterialParamsRef = MaterialParams::HostRef;
using BarnXs = units::BarnXs;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ElementSelectorTest : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = std::mt19937;
    //!@}

  protected:
    void SetUp() override
    {
        using namespace units;

        MaterialParams::Input inp;
        inp.elements = {
            {AtomicNumber{1}, AmuMass{1.008}, {}, "H"},
            {AtomicNumber{11}, AmuMass{22.98976928}, {}, "Na"},
            {AtomicNumber{13}, AmuMass{26.9815385}, {}, "Al"},
            {AtomicNumber{53}, AmuMass{126.90447}, {}, "I"},
        };
        inp.materials = {
            {0.0, 0.0, MatterState::unspecified, {}, "hard_vacuum"},
            {native_value_from(MolCcDensity{0.1}),
             293.0,
             MatterState::gas,
             {{ElementId{2}, 1.0}},
             "Al"},
            {native_value_from(MolCcDensity{0.05}),
             293.0,
             MatterState::solid,
             {{ElementId{0}, 0.25},
              {ElementId{1}, 0.25},
              {ElementId{2}, 0.25},
              {ElementId{3}, 0.25}},
             "everything_even"},
            {native_value_from(MolCcDensity{1}),
             293.0,
             MatterState::solid,
             {{ElementId{0}, 0.48},
              {ElementId{1}, 0.24},
              {ElementId{2}, 0.16},
              {ElementId{3}, 0.12}},
             "everything_weighted"},
        };
        mats = std::make_shared<MaterialParams>(std::move(inp));
        host_mats = mats->host_ref();

        // Allocate storage_
        storage_.assign(mats->max_element_components(), -1);
    }

    Span<real_type> storage() { return make_span(storage_); }

    Span<real_type> storage(size_type num_elements)
    {
        CELER_EXPECT(num_elements <= storage_.size());
        return this->storage().subspan(0, num_elements);
    }

    std::shared_ptr<MaterialParams> mats;
    MaterialParamsRef host_mats;
    RandomEngine rng;
    std::vector<real_type> storage_;
};

// Return cross section proportional to the element ID offset by 1.
auto mock_micro_xs(ElementId el_id) -> BarnXs
{
    CELER_EXPECT(el_id < 4);
    return BarnXs(static_cast<real_type>(el_id.get() + 1));
}

// Example functor for calculating cross section from actual atomic properties
// and particle state
struct CalcFancyMicroXs
{
    CalcFancyMicroXs(MaterialParamsRef const& mats, units::MevEnergy energy)
        : mats_(mats), inv_energy_(1 / energy.value())
    {
    }

    auto operator()(ElementId el_id) const -> BarnXs
    {
        CELER_EXPECT(el_id);
        ElementView el(mats_, el_id);
        return BarnXs(el.cbrt_z() * inv_energy_);
    }

    MaterialParamsRef const& mats_;
    real_type inv_energy_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

//! You can't select an element in pure void. (No interactions anyway.)
TEST_F(ElementSelectorTest, TEST_IF_CELERITAS_DEBUG(vacuum))
{
    MaterialView material(mats->host_ref(), mats->find_material("hard_vacuum"));
    EXPECT_THROW(ElementSelector(material, mock_micro_xs, this->storage()),
                 DebugError);
}

//! Single element should always select the first one.
TEST_F(ElementSelectorTest, single)
{
    MaterialView material(host_mats, mats->find_material("Al"));
    ElementSelector select_el(material, mock_micro_xs, this->storage());

    // Construction should have precalculated cross sections
    real_type const expected_elemental_micro_xs[] = {3};
    EXPECT_VEC_SOFT_EQ(expected_elemental_micro_xs,
                       this->storage(material.num_elements()));

    // Select a single element
    for ([[maybe_unused]] auto i : range(100))
    {
        auto el_id = select_el(rng);
        EXPECT_EQ(ElementComponentId{0}, el_id);
    }
}

//! Equal number densities but unequal cross sections
TEST_F(ElementSelectorTest, everything_even)
{
    MaterialView material(host_mats, mats->find_material("everything_even"));
    ElementSelector select_el(material, mock_micro_xs, this->storage());

    // Test cross sections
    real_type const expected_elemental_micro_xs[] = {1, 2, 3, 4};
    EXPECT_VEC_SOFT_EQ(expected_elemental_micro_xs,
                       this->storage(material.num_elements()));

    // Select a single element
    std::vector<int> tally(material.num_elements(), 0);
    for ([[maybe_unused]] auto i : range(10000))
    {
        auto el_id = select_el(rng);
        ASSERT_LT(el_id.get(), tally.size());
        ++tally[el_id.get()];
    }

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        // Proportional to micro_xs (equal number density)
        int const expected_tally[] = {1032, 2014, 2971, 3983};
        EXPECT_VEC_EQ(expected_tally, tally);
    }
}

//! Equal number densities but unequal cross sections
TEST_F(ElementSelectorTest, everything_even_seq)
{
    MaterialView material(host_mats, mats->find_material("everything_even"));
    ElementSelector select_el(material, mock_micro_xs, this->storage());

    // Test with sequence engine
    auto seq_rng = SequenceEngine::from_reals(
        {0.0, 0.099, 0.101, 0.3, 0.499, 0.999999});
    std::vector<int> selection;
    while (seq_rng.count() < seq_rng.max_count())
    {
        auto el_id = select_el(seq_rng);
        selection.push_back(el_id.unchecked_get());
    }
    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        int const expected_selection[] = {0, 0, 1, 2, 2, 3};
        EXPECT_VEC_EQ(expected_selection, selection);
    }
}

//! Number densities scaled to 1/xs so equiprobable
TEST_F(ElementSelectorTest, everything_weighted)
{
    MaterialView material(host_mats,
                          mats->find_material("everything_weighted"));
    ElementSelector select_el(material, mock_micro_xs, this->storage());

    // Test cross sections
    real_type const expected_elemental_micro_xs[] = {1, 2, 3, 4};
    EXPECT_VEC_SOFT_EQ(expected_elemental_micro_xs,
                       this->storage(material.num_elements()));

    // Select a single element
    std::vector<int> tally(material.num_elements(), 0);
    for ([[maybe_unused]] auto i : range(10000))
    {
        auto el_id = select_el(rng);
        ASSERT_LT(el_id.get(), tally.size());
        ++tally[el_id.get()];
    }

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        // Equiprobable
        int const expected_tally[] = {2574, 2395, 2589, 2442};
        EXPECT_VEC_EQ(expected_tally, tally);
    }
}

//! Many zero cross sections
TEST_F(ElementSelectorTest, even_zero_xs)
{
    MaterialView material(host_mats, mats->find_material("everything_even"));
    auto calc_xs = [](ElementId el) -> BarnXs {
        return BarnXs((el.get() % 2 ? 1 : 0));
    };
    ElementSelector select_el(material, calc_xs, this->storage());

    auto seq_rng = SequenceEngine::from_reals({0.0, 0.01, 0.49, 0.5, 0.51});
    std::vector<int> selection;
    while (seq_rng.count() < seq_rng.max_count())
    {
        auto el_id = select_el(seq_rng);
        selection.push_back(el_id.unchecked_get());
    }
    int const expected_selection[] = {1, 1, 1, 3, 3};
    EXPECT_VEC_EQ(expected_selection, selection);
}

//! Example of using a more complex/functional cross section functor
TEST_F(ElementSelectorTest, fancy_xs)
{
    units::MevEnergy energy{123};
    MaterialView material(host_mats,
                          mats->find_material("everything_weighted"));
    ElementSelector select_el(
        material, CalcFancyMicroXs{host_mats, energy}, this->storage());

    // Test cross sections
    real_type const expected_elemental_micro_xs[] = {
        0.008130081300813, 0.01808113894772, 0.01911654217659, 0.0305389085709};
    EXPECT_VEC_SOFT_EQ(expected_elemental_micro_xs,
                       this->storage(material.num_elements()));
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
