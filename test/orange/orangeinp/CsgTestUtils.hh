//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTestUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "geocel/Types.hh"
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

void print_expected(detail::CsgUnit const& u);
void print_expected(detail::IntersectSurfaceState const& css);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
