//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTypes.cc
//---------------------------------------------------------------------------//
#include "CsgTypes.hh"

#include <iostream>
#include <sstream>

#include "corecel/io/Join.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/univ/detail/Utils.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
//!@{
//! Write Node variants to a stream
std::ostream& operator<<(std::ostream& os, True const&)
{
    os << "true";
    return os;
}

std::ostream& operator<<(std::ostream& os, False const&)
{
    os << "false";
    return os;
}

std::ostream& operator<<(std::ostream& os, Aliased const& n)
{
    os << "->{" << n.node.unchecked_get() << '}';
    return os;
}

std::ostream& operator<<(std::ostream& os, Negated const& n)
{
    os << "not{" << n.node.unchecked_get() << '}';
    return os;
}

std::ostream& operator<<(std::ostream& os, Surface const& n)
{
    os << "surface " << n.id.unchecked_get();
    return os;
}

std::ostream& operator<<(std::ostream& os, Joined const& n)
{
    os << (n.op == op_and  ? "all"
           : n.op == op_or ? "any"
                           : "INVALID")
       << '{'
       << join(n.nodes.begin(),
               n.nodes.end(),
               ',',
               [](NodeId n) { return n.unchecked_get(); })
       << '}';
    return os;
}

std::ostream& operator<<(std::ostream& os, Node const& node)
{
    CELER_EXPECT(!node.valueless_by_exception());
    std::visit([&os](auto const& n) { os << n; }, node);
    return os;
}

//!@}
//---------------------------------------------------------------------------//
//! Convert a node variant to a string
std::string to_string(Node const& n)
{
    std::ostringstream os;
    os << n;
    return os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Construct from bottom/top z segments.
 */
SpecialTrapezoid::SpecialTrapezoid(ZSegment&& bot, ZSegment&& top)
    : bot_(std::move(bot)), top_(std::move(top))
{
    constexpr auto left = Bound::lo;
    constexpr auto right = Bound::hi;

    CELER_EXPECT(bot_.z < top_.z);

    // Calculate abs_tol_ based on extents
    auto r_min = std::min(bot_.r[left], top_.r[left]);
    auto r_max = std::max(bot_.r[right], top_.r[right]);
    Real3 const extents{r_max - r_min, top_.z - bot_.z, 0};
    abs_tol_ = ::celeritas::detail::BumpCalculator(
        Tolerance<>::from_default())(extents);

    // Determine variety
    SoftClose soft_close(abs_tol_);

    bool has_pointy_bot = soft_close(bot_.r[left], bot_.r[right]);
    bool has_pointy_top = soft_close(top_.r[left], top_.r[right]);

    CELER_EXPECT(!(has_pointy_bot && has_pointy_top));

    if (has_pointy_bot)
    {
        CELER_EXPECT(top_.r[left] < top_.r[right]);
        variety_ = Variety::pointy_bot;
    }
    else if (has_pointy_top)
    {
        CELER_EXPECT(bot_.r[left] < bot_.r[right]);
        variety_ = Variety::pointy_top;
    }
    else
    {
        CELER_EXPECT(top_.r[left] < top_.r[right]
                     && bot_.r[left] < bot_.r[right]);
        variety_ = Variety::quad;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the unique points in counterclockwise order, from the upper right.
 */
auto SpecialTrapezoid::unique_points() const -> VecReal2
{
    constexpr auto left = Bound::lo;
    constexpr auto right = Bound::hi;

    VecReal2 points;

    // Add top points
    points.push_back({top_.r[right], top_.z});
    if (variety_ != Variety::pointy_top)
    {
        points.push_back({top_.r[left], top_.z});
    }

    // Add bottom points
    points.push_back({bot_.r[left], bot_.z});
    if (variety_ != Variety::pointy_bot)
    {
        points.push_back({bot_.r[right], bot_.z});
    }

    return points;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
