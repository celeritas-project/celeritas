//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionGroups.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/ActionGroups.hh"

#include "ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

extern template class ActionGroups<CoreParams, CoreState>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
