//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Problem.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;

namespace inp
{
struct Problem;
}
struct ImportData;

namespace setup
{
//---------------------------------------------------------------------------//
// Set up the problem
std::shared_ptr<CoreParams>
problem(inp::Problem const& p, ImportData const& imported);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
