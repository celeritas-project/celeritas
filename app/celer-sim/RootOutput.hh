//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RootOutput.hh
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

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
struct RunnerInput;

//---------------------------------------------------------------------------//

// Store RunnerInput to ROOT file when ROOT is available
void write_to_root(RunnerInput const& cargs, RootFileManager* root_manager);

// Store CoreParams to ROOT file when ROOT is available
void write_to_root(CoreParams const& core_params,
                   RootFileManager* root_manager);

//---------------------------------------------------------------------------//

#if !CELERITAS_USE_ROOT
inline void write_to_root(RunnerInput const&, RootFileManager*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void write_to_root(CoreParams const&, RootFileManager*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
