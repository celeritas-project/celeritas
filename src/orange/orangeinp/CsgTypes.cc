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
}  // namespace orangeinp
}  // namespace celeritas
