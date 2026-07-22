//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Control.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
struct CoreSizes;
struct OpticalSizes;

namespace inp
{
struct CoreStateCapacity;
struct OpticalStateCapacity;
}  // namespace inp

namespace setup
{
//---------------------------------------------------------------------------//
// Resolve core state capacity values
CoreSizes capacity(inp::CoreStateCapacity const& c, size_type num_streams);
// Resolve optical state capacity values
OpticalSizes
capacity(inp::OpticalStateCapacity const& c, size_type num_streams);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
