//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTypes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <iosfwd>
#include <string>
#include <variant>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "corecel/math/HashUtils.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
//! Operator token
using OperatorToken = logic::OperatorToken;

//! Unique identifier for a node
using NodeId = OpaqueId<struct Node_>;

inline constexpr OperatorToken op_and = OperatorToken::land;
inline constexpr OperatorToken op_or = OperatorToken::lor;

//! Node that represents "always true" for simplification
struct True
{
};

//! Node that represents "always false" for simplification
struct False
{
};

//! Node that is a single surface
struct Surface
{
    LocalSurfaceId id;
};

//! Node that negates the next ID
struct Negated
{
    NodeId node;
};

//! (Internal) stand-in node for a replacement for another node ID
struct Aliased
{
    NodeId node;
};

//! Internal node applying an operation to multiple leaf nodes
struct Joined
{
    OperatorToken op;
    std::vector<NodeId> nodes;
};

//! Generic node
using Node = std::variant<True, False, Surface, Negated, Aliased, Joined>;

//---------------------------------------------------------------------------//
// Equality operators
//---------------------------------------------------------------------------//
inline constexpr bool operator==(True const&, True const&)
{
    return true;
}
inline constexpr bool operator==(False const&, False const&)
{
    return true;
}
inline constexpr bool operator==(Surface const& a, Surface const& b)
{
    return a.id == b.id;
}
inline constexpr bool operator==(Negated const& a, Negated const& b)
{
    return a.node == b.node;
}
inline constexpr bool operator==(Aliased const& a, Aliased const& b)
{
    return a.node == b.node;
}
inline constexpr bool operator==(Joined const& a, Joined const& b)
{
    return a.op == b.op && a.nodes == b.nodes;
}

#define CELER_DEFINE_CSG_NE(CLS)                                 \
    inline constexpr bool operator!=(CLS const& a, CLS const& b) \
    {                                                            \
        return !(a == b);                                        \
    }

CELER_DEFINE_CSG_NE(True)
CELER_DEFINE_CSG_NE(False)
CELER_DEFINE_CSG_NE(Surface)
CELER_DEFINE_CSG_NE(Negated)
CELER_DEFINE_CSG_NE(Aliased)
CELER_DEFINE_CSG_NE(Joined)

//---------------------------------------------------------------------------//
// Stream output
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, True const&);
std::ostream& operator<<(std::ostream& os, False const&);
std::ostream& operator<<(std::ostream& os, Surface const&);
std::ostream& operator<<(std::ostream& os, Negated const&);
std::ostream& operator<<(std::ostream& os, Aliased const&);
std::ostream& operator<<(std::ostream& os, Joined const&);

// Write a variant node to a stream
std::ostream& operator<<(std::ostream& os, Node const&);

// Get a string representation of a variant node
std::string to_string(Node const&);

//---------------------------------------------------------------------------//
// Helper functions
//---------------------------------------------------------------------------//
//! Whether a node is a boolean (True or False instances)
inline constexpr bool is_boolean_node(Node const& n)
{
    return n.index() < 2;
}

//---------------------------------------------------------------------------//
//! Whether a node is a literal (true, false, surface)
inline constexpr bool is_literal_node(Node const& n)
{
    return n.index() < 3;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas

namespace std
{
//---------------------------------------------------------------------------//
// HASH SPECIALIZATIONS
//---------------------------------------------------------------------------//
//! \cond
template<>
struct hash<celeritas::orangeinp::True>
{
    using argument_type = celeritas::orangeinp::True;
    using result_type = std::size_t;
    result_type operator()(argument_type const&) const noexcept
    {
        return result_type{0};
    }
};

template<>
struct hash<celeritas::orangeinp::False>
{
    using argument_type = celeritas::orangeinp::False;
    using result_type = std::size_t;
    result_type operator()(argument_type const&) const noexcept
    {
        return result_type{0};
    }
};

template<>
struct hash<celeritas::orangeinp::Surface>
{
    using argument_type = celeritas::orangeinp::Surface;
    using result_type = std::size_t;
    result_type operator()(argument_type const& val) const noexcept
    {
        return std::hash<celeritas::LocalSurfaceId>{}(val.id);
    }
};

template<>
struct hash<celeritas::orangeinp::Negated>
{
    using argument_type = celeritas::orangeinp::Negated;
    using result_type = std::size_t;
    result_type operator()(argument_type const& val) const noexcept
    {
        return std::hash<celeritas::orangeinp::NodeId>{}(val.node);
    }
};

template<>
struct hash<celeritas::orangeinp::Aliased>
{
    using argument_type = celeritas::orangeinp::Aliased;
    using result_type = std::size_t;
    result_type operator()(argument_type const& val) const noexcept
    {
        return std::hash<celeritas::orangeinp::NodeId>{}(val.node);
    }
};

template<>
struct hash<celeritas::orangeinp::Joined>
{
    using argument_type = celeritas::orangeinp::Joined;
    using result_type = std::size_t;
    result_type operator()(argument_type const& val) const noexcept
    {
        result_type result;
        celeritas::Hasher hash{&result};
        hash(static_cast<std::size_t>(val.op));
        hash(val.nodes.size());
        for (auto& v : val.nodes)
        {
            hash(std::hash<celeritas::orangeinp::NodeId>{}(v));
        }
        return result;
    }
};
//! \endcond
//---------------------------------------------------------------------------//
}  // namespace std
