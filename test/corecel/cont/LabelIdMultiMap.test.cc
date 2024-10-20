//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/LabelIdMultiMap.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/LabelIdMultiMap.hh"

#include <iostream>
#include <sstream>

#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//

using CatId = OpaqueId<struct Cat_>;
using CatMultiMap = LabelIdMultiMap<CatId>;
using VecLabel = CatMultiMap::VecLabel;

std::ostream& operator<<(std::ostream& os, CatId const& cat)
{
    os << "CatId{";
    if (cat)
        os << cat.unchecked_get();
    os << "}";
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace

TEST(LabelIdMultiMapTest, empty)
{
    CatMultiMap const default_cats;
    EXPECT_FALSE(default_cats);  // Uninitialized and empty
    CatMultiMap const empty_cats("cat", VecLabel{});
    EXPECT_TRUE(empty_cats);  // Initialized and empty
    for (CatMultiMap const* cats : {&default_cats, &empty_cats})
    {
        EXPECT_EQ(0, cats->size());
        EXPECT_EQ(CatId{}, cats->find(Label{"merry"}));
        EXPECT_EQ(0, cats->find_all("pippin").size());
        if (CELERITAS_DEBUG)
        {
            EXPECT_THROW(cats->at(CatId{0}), DebugError);
        }
    }
}

TEST(LabelIdMultiMapTest, no_ext_with_duplicates)
{
    CatMultiMap cats{
        "cat", VecLabel{{"dexter", "andy", "loki", "bob", "bob", "bob"}}};
    EXPECT_TRUE(cats);
    EXPECT_EQ(6, cats.size());
    EXPECT_EQ(CatId{}, cats.find("nyoka"));
    EXPECT_EQ(CatId{0}, cats.find("dexter"));
    EXPECT_EQ(CatId{1}, cats.find("andy"));
    EXPECT_EQ(CatId{2}, cats.find("loki"));
    EXPECT_EQ(CatId{2}, cats.find(Label{"loki"}));

    EXPECT_EQ(CatId{}, cats.find_unique("nyoka"));
    EXPECT_EQ(CatId{2}, cats.find_unique("loki"));
    EXPECT_THROW(cats.find_unique("bob"), RuntimeError);

    static CatId const expected_duplicates[] = {CatId{3}, CatId{4}, CatId{5}};
    EXPECT_VEC_EQ(expected_duplicates, cats.duplicates());
}

TEST(LabelIdMultiMapTest, empty_duplicates)
{
    CatMultiMap cats{VecLabel{{"dexter", "andy", "loki", "", ""}}};
    EXPECT_TRUE(cats);
    EXPECT_EQ(5, cats.size());
    EXPECT_EQ(0, cats.duplicates().size());
}

TEST(LabelIdMultiMapTest, some_labels)
{
    CatMultiMap cats{{{Label{"leroy"},
                       Label{"fluffy"},
                       {"fluffy", "jr"},
                       {"fluffy", "sr"}}}};
    EXPECT_EQ(4, cats.size());
    EXPECT_EQ(CatId{1}, cats.find(Label{"fluffy"}));
    EXPECT_EQ(CatId{2}, cats.find(Label{"fluffy", "jr"}));
    {
        auto found = cats.find_all("fluffy");
        static CatId const expected_found[] = {CatId{1}, CatId{2}, CatId{3}};
        EXPECT_VEC_EQ(expected_found, found);
    }

    EXPECT_THROW(cats.find_unique("fluffy"), RuntimeError);
}

TEST(LabelIdMultiMapTest, shuffled_labels)
{
    std::vector<Label> const labels = {
        {"c", "2"},
        {"b", "1"},
        {"b", "0"},
        {"a", "0"},
        {"c", "0"},
        {"c", "1"},
    };

    CatMultiMap cats{std::vector<Label>{labels}};

    // Check ordering of IDs
    for (auto i : range<CatId::size_type>(labels.size()))
    {
        EXPECT_EQ(labels[i], cats.at(CatId{i}));
    }

    // Check discontinuous ID listing
    {
        auto found = cats.find_all("a");
        static CatId const expected_found[] = {CatId{3}};
        EXPECT_VEC_EQ(expected_found, found);
    }
    {
        auto found = cats.find_all("b");
        static CatId const expected_found[] = {CatId{2}, CatId{1}};
        EXPECT_VEC_EQ(expected_found, found);
    }
    {
        auto found = cats.find_all("c");
        static CatId const expected_found[] = {CatId{4}, CatId{5}, CatId{0}};
        EXPECT_VEC_EQ(expected_found, found);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
