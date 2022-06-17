//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/LabelIdMultiMap.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/LabelIdMultiMap.hh"

#include <iostream>

#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

using celeritas::Label;
using celeritas::LabelIdMultiMap;

using CatId       = celeritas::OpaqueId<struct Cat>;
using CatMultiMap = LabelIdMultiMap<CatId>;

std::ostream& operator<<(std::ostream& os, const CatId& cat)
{
    os << "CatId{";
    if (cat)
        os << cat.unchecked_get();
    os << "}";
    return os;
}

TEST(LabelCmp, ordering)
{
    EXPECT_EQ(Label("a"), Label("a"));
    EXPECT_EQ(Label("a", "1"), Label("a", "1"));
    EXPECT_NE(Label("a"), Label("b"));
    EXPECT_NE(Label("a", "1"), Label("a", "2"));
    EXPECT_TRUE(Label("a") < Label("b"));
    EXPECT_FALSE(Label("a") < Label("a"));
    EXPECT_FALSE(Label("b") < Label("a"));
    EXPECT_TRUE(Label("a") < Label("a", "1"));
    EXPECT_TRUE(Label("a", "0") < Label("a", "1"));
    EXPECT_FALSE(Label("a", "1") < Label("a", "1"));
    EXPECT_FALSE(Label("a", "2") < Label("a", "1"));
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class LabelIdMultiMapTest : public celeritas_test::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LabelIdMultiMapTest, exceptions)
{
    EXPECT_THROW(CatMultiMap({Label{"kali"}, Label{"kali"}}),
                 celeritas::RuntimeError);
#if CELERITAS_DEBUG
    EXPECT_THROW(CatMultiMap({}), celeritas::DebugError);
#endif
}

TEST_F(LabelIdMultiMapTest, no_labels)
{
    CatMultiMap cats{{Label{"dexter"}, Label{"andy"}, Label{"loki"}}};
    EXPECT_EQ(3, cats.size());
    EXPECT_EQ(CatId{}, cats.find(Label{"nyoka"}));
    EXPECT_EQ(CatId{0}, cats.find(Label{"dexter"}));
    EXPECT_EQ(CatId{1}, cats.find(Label{"andy"}));
    EXPECT_EQ(CatId{2}, cats.find(Label{"loki"}));
}

TEST_F(LabelIdMultiMapTest, some_labels)
{
    CatMultiMap cats{{{Label{"leroy"},
                       Label{"fluffy"},
                       {"fluffy", "jr"},
                       {"fluffy", "sr"}}}};
    EXPECT_EQ(4, cats.size());
    EXPECT_EQ(CatId{1}, cats.find(Label{"fluffy"}));
    EXPECT_EQ(CatId{2}, cats.find(Label{"fluffy", "jr"}));
    {
        auto               found            = cats.find("fluffy");
        static const CatId expected_found[] = {CatId{1}, CatId{2}, CatId{3}};
        EXPECT_VEC_EQ(expected_found, found);
    }
}

TEST_F(LabelIdMultiMapTest, shuffled_labels)
{
    const std::vector<Label> labels = {
        {"c", "2"},
        {"b", "1"},
        {"b", "0"},
        {"a", "0"},
        {"c", "0"},
        {"c", "1"},
    };

    CatMultiMap cats{labels};

    // Check ordering of IDs
    for (auto i : celeritas::range(labels.size()))
    {
        EXPECT_EQ(labels[i], cats.get(CatId{i}));
    }

    // Check discontinuous ID listing
    {
        auto               found            = cats.find("a");
        static const CatId expected_found[] = {CatId{3}};
        EXPECT_VEC_EQ(expected_found, found);
    }
    {
        auto               found            = cats.find("b");
        static const CatId expected_found[] = {CatId{2}, CatId{1}};
        EXPECT_VEC_EQ(expected_found, found);
    }
    {
        auto               found            = cats.find("c");
        static const CatId expected_found[] = {CatId{4}, CatId{5}, CatId{0}};
        EXPECT_VEC_EQ(expected_found, found);
    }
}
