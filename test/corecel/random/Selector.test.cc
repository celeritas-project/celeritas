//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Selector.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/Selector.hh"

#include "corecel/OpaqueId.hh"
#include "corecel/cont/EnumArray.hh"

#include "SequenceEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
SequenceEngine make_rng(double select_val)
{
    return SequenceEngine::from_reals({select_val});
}

using SelectorTest = Test;

TEST_F(SelectorTest, typical)
{
    static double const prob[] = {0.1, 0.3, 0.5, 0.1};

    auto sample_prob = make_selector([](int i) { return prob[i]; }, 4, 1.0);
    auto rng = make_rng(0.0);
    EXPECT_TRUE((std::is_same<decltype(sample_prob(rng)), int>::value));

    rng = make_rng(0.0);
    EXPECT_EQ(0, sample_prob(rng));

    rng = make_rng(0.0999);
    EXPECT_EQ(0, sample_prob(rng));

    rng = make_rng(0.1001);
    EXPECT_EQ(1, sample_prob(rng));

    rng = make_rng(0.4001);
    EXPECT_EQ(2, sample_prob(rng));

    rng = make_rng(0.9001);
    EXPECT_EQ(3, sample_prob(rng));

    // Check that highest representable value doesn't go off the end
    rng = SequenceEngine{{0xffffffffu, 0xffffffffu}};
    EXPECT_EQ(3, sample_prob(rng));
}

TEST_F(SelectorTest, zeros)
{
    static double const prob[] = {0.0, 0.0, 0.4, 0.6};

    auto sample_prob = make_selector([](int i) { return prob[i]; }, 4, 1.0);

    auto rng = make_rng(0.0);
    EXPECT_EQ(2, sample_prob(rng));

    rng = make_rng(1e-15);
    EXPECT_EQ(2, sample_prob(rng));
}

TEST_F(SelectorTest, TEST_IF_CELERITAS_DEBUG(errors))
{
    std::vector<double> prob = {0.1, 0.3, 0.5, 0.1};
    auto get_val = [&prob](std::size_t i) {
        CELER_ASSERT(i < prob.size());
        return prob[i];
    };

    EXPECT_THROW(make_selector(get_val, prob.size(), 1.1), DebugError);
    EXPECT_THROW(make_selector(get_val, prob.size(), 0.9), DebugError);

    prob.push_back(-0.1);
    EXPECT_THROW(make_selector(get_val, prob.size(), 0.9), DebugError);
}

TEST_F(SelectorTest, make_selector)
{
    static double const prob[] = {0.1, 0.3, 0.5, 0.1};

    auto sample_prob
        = make_selector([](int i) { return prob[i]; }, std::size(prob));

    auto rng = make_rng(0.0);
    EXPECT_EQ(0, sample_prob(rng));

    rng = make_rng(0.999999);
    EXPECT_EQ(3, sample_prob(rng));
}

TEST_F(SelectorTest, selector_element)
{
    using ElementId = OpaqueId<struct Element_>;
    static double const macro_xs[] = {1.0, 2.0, 4.0};
    std::vector<int> evaluated;
    auto get_xs = [&evaluated](ElementId el) {
        CELER_EXPECT(el < 3);
        evaluated.push_back(el.get());
        return macro_xs[el.get()];
    };

    auto sample_el = make_selector(get_xs, ElementId{3}, 1 + 2 + 4);
    auto rng = make_rng(0.0);
    EXPECT_TRUE((std::is_same<decltype(sample_el(rng)), ElementId>::value));

    rng = make_rng(0.0);
    EXPECT_EQ(0, sample_el(rng).unchecked_get());

    rng = make_rng(0.9999 / 7.0);
    EXPECT_EQ(0, sample_el(rng).unchecked_get());

    rng = make_rng(1.000000001 / 7.0);
    EXPECT_EQ(1, sample_el(rng).unchecked_get());

    rng = make_rng(3.0001 / 7.0);
    EXPECT_EQ(2, sample_el(rng).unchecked_get());

    // In debug, extra error checking evaluates all IDs during construction.
    // Final value is only ever evaluated as part of debugging.
    if (CELERITAS_DEBUG)
    {
        int const expected_evaluated_final[] = {0, 1, 2, 0, 0, 0, 1, 0, 1};
        EXPECT_VEC_EQ(expected_evaluated_final, evaluated);
    }
    else
    {
        int const expected_evaluated_final[] = {0, 0, 0, 1, 0, 1};
        EXPECT_VEC_EQ(expected_evaluated_final, evaluated);
    }
}

TEST_F(SelectorTest, selector_enum)
{
    enum class Interaction
    {
        scatter,
        fission,
        gamma,
        unknown,
        size_
    };
    EnumArray<Interaction, double> macro_xs{0.1, 0.3, 0.5, 0.1};

    auto sample_xs = make_selector(
        [&macro_xs](Interaction i) { return macro_xs[i]; }, Interaction::size_);

    auto rng = make_rng(0.00001);
    EXPECT_EQ(Interaction::scatter, sample_xs(rng));

    rng = make_rng(0.999999);
    EXPECT_EQ(Interaction::unknown, sample_xs(rng));
}

//---------------------------------------------------------------------------//

using UnnormSelectorTest = Test;

TEST_F(UnnormSelectorTest, make_selector)
{
    static double const prob[] = {0.1, 0.3, 0.5};

    auto sample_prob = make_unnormalized_selector(
        [](int i) { return prob[i]; }, std::size(prob), real_type{1});

    auto rng = make_rng(0.0);
    EXPECT_EQ(0, sample_prob(rng));

    rng = make_rng(0.999999);
    EXPECT_EQ(3, sample_prob(rng));
}

TEST_F(UnnormSelectorTest, selector_enum)
{
    enum class Interaction
    {
        scatter,
        fission,
        gamma,
        size_
    };
    EnumArray<Interaction, double> macro_xs{0.1, 0.3, 0.5};

    auto sample_xs = make_unnormalized_selector(
        [&macro_xs](Interaction i) { return macro_xs[i]; },
        Interaction::size_,
        1.0);

    auto rng = make_rng(0.00001);
    EXPECT_EQ(Interaction::scatter, sample_xs(rng));

    rng = make_rng(0.999999);
    EXPECT_EQ(Interaction::size_, sample_xs(rng));
}

TEST_F(UnnormSelectorTest, TEST_IF_CELERITAS_DEBUG(errors))
{
    std::vector<double> prob = {0.1, 0.3, 0.5, 0.1};
    auto get_val = [&prob](std::size_t i) {
        CELER_ASSERT(i < prob.size());
        return prob[i];
    };

    EXPECT_THROW(make_selector(get_val, prob.size(), 0.8), DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
