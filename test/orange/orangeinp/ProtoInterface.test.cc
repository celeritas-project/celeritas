//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ProtoInterface.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/ProtoInterface.hh"

#include "corecel/io/Join.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
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
    void build(GlobalBuilder&) const { CELER_ASSERT_UNREACHABLE(); }

  private:
    std::string label_;
    VecProto daughters_;
};

//---------------------------------------------------------------------------//
class ProtoInterfaceTest : public ::celeritas::test::Test
{
  protected:
    using VecProto = ProtoInterface::VecProto;

    std::string proto_labels(VecProto const& vp)
    {
        return to_string(
            join_stream(vp.begin(),
                        vp.end(),
                        ",",
                        [](std::ostream& os, ProtoInterface const* p) {
                            if (!p)
                            {
                                os << "<null>";
                            }
                            else
                            {
                                os << p->label();
                            }
                        }));
    }
};

TEST_F(ProtoInterfaceTest, all)
{
    TestProto const e{"e", {&e}};
    TestProto const d{"d", {}};
    TestProto const b{"b", {&d, &d, &d}};
    TestProto const c{"c", {&d, &b, &e}};
    TestProto const global{"a", {&b, &c}};

    EXPECT_EQ("a,b,c,d,e", proto_labels(build_ordering(global)));
    EXPECT_EQ("c,d,b,e", proto_labels(build_ordering(c)));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
