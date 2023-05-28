//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/RunnerInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
class RootFileManager;
//---------------------------------------------------------------------------//
}  // namespace celeritas

namespace demo_loop
{
//---------------------------------------------------------------------------//
struct RunnerInput;

//---------------------------------------------------------------------------//

// Store RunnerInput to ROOT file when ROOT is available
void write_to_root(RunnerInput const& cargs,
                   celeritas::RootFileManager* root_manager);

// Store CoreParams to ROOT file when ROOT is available
void write_to_root(celeritas::CoreParams const& core_params,
                   celeritas::RootFileManager* root_manager);

//---------------------------------------------------------------------------//

#if !CELERITAS_USE_ROOT
inline void write_to_root(RunnerInput const&, celeritas::RootFileManager*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void
write_to_root(celeritas::CoreParams const&, celeritas::RootFileManager*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace demo_loop
