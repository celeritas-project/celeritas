//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTestUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <string>

#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class CsgTree;

namespace detail
{
struct BoundingZone;
struct IntersectSurfaceState;
struct CsgUnit;
}  // namespace detail

namespace test
{
//---------------------------------------------------------------------------//
std::string to_json_string(CsgTree const&);

std::vector<int> to_vec_int(std::vector<NodeId> const& nodes);
std::vector<std::string> surface_strings(detail::CsgUnit const& u);
std::vector<std::string> volume_strings(detail::CsgUnit const& u);
std::string tree_string(detail::CsgUnit const& u);
std::vector<std::string> md_strings(detail::CsgUnit const& u);
std::vector<std::string> bound_strings(detail::CsgUnit const& u);
std::vector<std::string> transform_strings(detail::CsgUnit const& u);
std::vector<int> volume_nodes(detail::CsgUnit const& u);
std::vector<std::string> fill_strings(detail::CsgUnit const& u);
std::vector<real_type> flattened(detail::BoundingZone const& bz);

std::string count_surface_types(detail::CsgUnit const& u);

void print_expected(detail::CsgUnit const& u);
void print_expected(detail::IntersectSurfaceState const& css);

// Functions for `join_stream`
void stream_node_id(std::ostream& os, NodeId n);
void stream_logic_int(std::ostream& os, logic_int value);

//! Pretty-print a vector of logic
struct ReprLogic
{
    std::vector<logic_int> const& logic;
};

std::ostream& operator<<(std::ostream& os, ReprLogic const& rl);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
