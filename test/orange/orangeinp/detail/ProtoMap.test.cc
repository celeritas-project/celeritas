//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ProtoMap.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/ProtoMap.hh"

#include "corecel/io/Join.hh"
#include "orange/orangeinp/ProtoInterface.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
class TestProto : public ProtoInterface
{
  public:
    TestProto(std::string label, VecProto&& daughters)
        : label_{label}, daughters_{daughters}
    {
    }

    //! Short unique name of this object
    std::string_view label() const { return label_; }

    //! Get the boundary of this universe as an object
    SPConstObject interior() const { CELER_ASSERT_UNREACHABLE(); }

    //! Get a non-owning set of all daughters referenced by this proto
    VecProto daughters() const { return daughters_; }

    //! Construct a universe input from this object
    void build(InputBuilder&) const { CELER_ASSERT_UNREACHABLE(); }

  private:
    std::string label_;
    VecProto daughters_;
};

//---------------------------------------------------------------------------//
class ProtoMapTest : public ::celeritas::test::Test
{
};

TEST_F(ProtoMapTest, deep_and_wide)
{
    TestProto const g{"g", {}};
    TestProto const f{"f", {}};
    TestProto const e{"e", {&g}};
    TestProto const d{"d", {}};
    TestProto const b{"b", {&d, &d, &d}};
    TestProto const c{"c", {&d, &b, &e, &f}};
    TestProto const a{"a", {&b, &c}};

    ProtoMap pm(a);
    ASSERT_EQ(7, pm.size());
    EXPECT_EQ("a", pm.at(UniverseId{0})->label());
    EXPECT_EQ("b", pm.at(UniverseId{1})->label());
    EXPECT_EQ("c", pm.at(UniverseId{2})->label());
    EXPECT_EQ("d", pm.at(UniverseId{3})->label());
    EXPECT_EQ("e", pm.at(UniverseId{4})->label());
    EXPECT_EQ("f", pm.at(UniverseId{5})->label());
    EXPECT_EQ("g", pm.at(UniverseId{6})->label());

    EXPECT_EQ(UniverseId{0}, pm.find(&a));
    EXPECT_EQ(UniverseId{1}, pm.find(&b));
    EXPECT_EQ(UniverseId{2}, pm.find(&c));
    EXPECT_EQ(UniverseId{3}, pm.find(&d));
    EXPECT_EQ(UniverseId{4}, pm.find(&e));
    EXPECT_EQ(UniverseId{5}, pm.find(&f));
    EXPECT_EQ(UniverseId{6}, pm.find(&g));

    if (CELERITAS_DEBUG)
    {
        TestProto const none{"none", {}};
        EXPECT_THROW(pm.find(&none), DebugError);
        EXPECT_THROW(pm.at(UniverseId{7}), DebugError);
    }
}

TEST_F(ProtoMapTest, asymmetric)
{
    TestProto const e{"e", {}};
    TestProto const d{"d", {}};
    TestProto const c{"c", {}};
    TestProto const b{"b", {&d, &e}};
    TestProto const a{"a", {&b, &c}};

    ProtoMap pm(a);
    ASSERT_EQ(5, pm.size());
    EXPECT_EQ("a", pm.at(UniverseId{0})->label());
    EXPECT_EQ("b", pm.at(UniverseId{1})->label());
    EXPECT_EQ("c", pm.at(UniverseId{2})->label());
    EXPECT_EQ("d", pm.at(UniverseId{3})->label());
    EXPECT_EQ("e", pm.at(UniverseId{4})->label());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
