//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/GeoSetup.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/GeoSetup.hh"

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
    void build(GeoSetup const&, BuildResult*) const
    {
        CELER_ASSERT_UNREACHABLE();
    }

  private:
    std::string label_;
    VecProto daughters_;
};

//---------------------------------------------------------------------------//
class GeoSetupTest : public ::celeritas::test::Test
{
  protected:
    using Tol = Tolerance<>;

    Tolerance<> tol_ = Tol::from_relative(1e-5);
};

TEST_F(GeoSetupTest, s)
{
    TestProto const g{"g", {}};
    TestProto const f{"f", {}};
    TestProto const e{"e", {&g}};
    TestProto const d{"d", {}};
    TestProto const b{"b", {&d, &d, &d}};
    TestProto const c{"c", {&d, &b, &e, &f}};
    TestProto const a{"a", {&b, &c}};

    {
        GeoSetup gs(tol_, a);
        ASSERT_EQ(7, gs.size());
        EXPECT_EQ("a", gs.at(UniverseId{0})->label());
        EXPECT_EQ("b", gs.at(UniverseId{1})->label());
        EXPECT_EQ("c", gs.at(UniverseId{2})->label());
        EXPECT_EQ("d", gs.at(UniverseId{3})->label());
        EXPECT_EQ("e", gs.at(UniverseId{4})->label());
        EXPECT_EQ("f", gs.at(UniverseId{5})->label());
        EXPECT_EQ("g", gs.at(UniverseId{6})->label());

        EXPECT_EQ(UniverseId{0}, gs.find(&a));
        EXPECT_EQ(UniverseId{1}, gs.find(&b));
        EXPECT_EQ(UniverseId{2}, gs.find(&c));
        EXPECT_EQ(UniverseId{3}, gs.find(&d));
        EXPECT_EQ(UniverseId{4}, gs.find(&e));
        EXPECT_EQ(UniverseId{5}, gs.find(&f));
        EXPECT_EQ(UniverseId{6}, gs.find(&g));

        if (CELERITAS_DEBUG)
        {
            TestProto const none{"none", {}};
            EXPECT_THROW(gs.find(&none), DebugError);
            EXPECT_THROW(gs.at(UniverseId{7}), DebugError);
        }
    }
    {
        GeoSetup gs(tol_, c);
        ASSERT_EQ(6, gs.size());
        EXPECT_EQ("c", gs.at(UniverseId{0})->label());
        EXPECT_EQ("d", gs.at(UniverseId{1})->label());
        EXPECT_EQ("b", gs.at(UniverseId{2})->label());
        EXPECT_EQ("e", gs.at(UniverseId{3})->label());
        EXPECT_EQ("f", gs.at(UniverseId{4})->label());
        EXPECT_EQ("g", gs.at(UniverseId{5})->label());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
