//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootCoreParamsOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
class ActionRegistry;
class RootFileManager;
struct GeantPhysicsOptions;

//---------------------------------------------------------------------------//

// Store actions to ROOT file
void write_to_root(ActionRegistry const& actions,
                   RootFileManager* root_manager);

//---------------------------------------------------------------------------//

#if !CELERITAS_USE_ROOT
inline void write_to_root(ActionRegistry const&, RootFileManager*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
