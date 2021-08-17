//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoRun.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "LDemoIO.hh"

using celeritas::MemSpace;

namespace demo_loop
{
//---------------------------------------------------------------------------//
template<MemSpace M>
LDemoResult run_demo(LDemoArgs args);

//---------------------------------------------------------------------------//
} // namespace demo_loop

#include "LDemoRun.i.hh"
