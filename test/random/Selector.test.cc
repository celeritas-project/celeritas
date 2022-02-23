//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Selector.test.cc
//---------------------------------------------------------------------------//
#include "random/Selector.hh"

#include "base/OpaqueId.hh"

#include "SequenceEngine.hh"
#include "celeritas_test.hh"

using celeritas::make_selector;
using celeritas_test::SequenceEngine;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PdfSelectorTest : public celeritas::Test
{
  public:
    using SelectorT = celeritas::Selector<std::function<double(int)>, int>;
};

SequenceEngine make_rng(double select_val)
{
    return SequenceEngine::from_reals({select_val});
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PdfSelectorTest, typical)
{
    static const double prob[] = {0.1, 0.3, 0.5, 0.1};

    SelectorT sample_prob{[](int i) { return prob[i]; }, 4, 1.0};
    auto      rng = make_rng(0.0);
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

TEST_F(PdfSelectorTest, zeros)
{
    static const double prob[] = {0.0, 0.0, 0.4, 0.6};

    SelectorT sample_prob{[](int i) { return prob[i]; }, 4, 1.0};

    auto rng = make_rng(0.0);
    EXPECT_EQ(2, sample_prob(rng));

    rng = make_rng(1e-15);
    EXPECT_EQ(2, sample_prob(rng));
}

TEST_F(PdfSelectorTest, TEST_IF_CELERITAS_DEBUG(invalid_total))
{
    static const double prob[]  = {0.1, 0.3, 0.5, 0.1};
    auto                get_val = [](int i) { return prob[i]; };

    EXPECT_THROW(SelectorT(get_val, 4, 1.1), celeritas::DebugError);
    EXPECT_THROW(SelectorT(get_val, 4, 0.9), celeritas::DebugError);
}

TEST(SelectorTest, make_selector)
{
    static const double prob[] = {0.1, 0.3, 0.5, 0.1};

    auto sample_prob = make_selector([](int i) { return prob[i]; }, 4);

    auto rng = make_rng(0.0);
    EXPECT_EQ(0, sample_prob(rng));

    rng = make_rng(0.999999);
    EXPECT_EQ(3, sample_prob(rng));
}

TEST(SelectorTest, selector_element)
{
    using ElementId                = celeritas::OpaqueId<struct Element>;
    static const double macro_xs[] = {1.0, 2.0, 4.0};
    std::vector<int>    evaluated;
    auto                get_xs = [&evaluated](ElementId el) {
        CELER_EXPECT(el < 3);
        evaluated.push_back(el.get());
        return macro_xs[el.get()];
    };

    auto sample_el = make_selector(get_xs, ElementId{3}, 1 + 2 + 4);
    auto rng       = make_rng(0.0);
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
        const int expected_evaluated_final[] = {0, 1, 2, 0, 0, 0, 1, 0, 1};
        EXPECT_VEC_EQ(expected_evaluated_final, evaluated);
    }
    else
    {
        const int expected_evaluated_final[] = {0, 0, 0, 1, 0, 1};
        EXPECT_VEC_EQ(expected_evaluated_final, evaluated);
    }
}
