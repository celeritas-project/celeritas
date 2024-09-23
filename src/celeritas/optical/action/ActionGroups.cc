//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/ActionGroups.cc
//---------------------------------------------------------------------------//
#include "ActionGroups.hh"

#include "corecel/sys/ActionGroups.t.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template class ActionGroups<optical::CoreParams, optical::CoreState>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
