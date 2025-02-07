//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionGroups.cc
//---------------------------------------------------------------------------//
#include "ActionGroups.hh"

#include "corecel/sys/ActionGroups.t.hh"

#include "CoreParams.hh"
#include "CoreState.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class ActionGroups<CoreParams, CoreState>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
