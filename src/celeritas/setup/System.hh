//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/System.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace inp
{
struct System;
}

namespace setup
{
//---------------------------------------------------------------------------//
// Set up low level system properties
void system(inp::System const& sys);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
