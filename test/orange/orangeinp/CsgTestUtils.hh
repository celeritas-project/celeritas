//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTestUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "orange/orangeinp/CsgTypes.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace orangeinp
{
class CsgTree;

namespace test
{
//---------------------------------------------------------------------------//
std::string to_json_string(CsgTree const&);

//---------------------------------------------------------------------------//
}  // namespace test

namespace detail
{
struct BoundingZone;
struct ConvexSurfaceState;
struct CsgUnit;

namespace test
{
//---------------------------------------------------------------------------//

std::vector<int> to_vec_int(std::vector<NodeId> const& nodes);
std::vector<std::string> surface_strings(CsgUnit const& u);
std::string tree_string(CsgUnit const& u);
std::vector<std::string> md_strings(CsgUnit const& u);
std::vector<real_type> flattened_bboxes(CsgUnit const& u);
std::vector<int> volume_nodes(CsgUnit const& u);
std::vector<std::string> fill_strings(CsgUnit const& u);
std::vector<real_type> flattened(BoundingZone const& bz);

void print_expected(CsgUnit const& u);
void print_expected(ConvexSurfaceState const& css);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
