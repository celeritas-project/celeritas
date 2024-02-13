//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/CsgTestUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/Types.hh"

namespace celeritas
{
class CsgTree;

namespace test
{
//---------------------------------------------------------------------------//
std::string to_json_string(CsgTree const&);

//---------------------------------------------------------------------------//
}  // namespace test

namespace orangeinp
{
namespace detail
{
struct CsgUnit;

namespace test
{
//---------------------------------------------------------------------------//

std::vector<std::string> surface_strings(CsgUnit const& u);
std::string tree_string(CsgUnit const& u);
std::vector<std::string> md_strings(CsgUnit const& u);
std::vector<real_type> flattened_bboxes(CsgUnit const& u);
std::vector<int> volume_nodes(CsgUnit const& u);
std::vector<std::string> fill_strings(CsgUnit const& u);

void print_expected(CsgUnit const& u);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
