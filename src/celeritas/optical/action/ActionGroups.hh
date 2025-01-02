//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/ActionGroups.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/ActionGroups.hh"

#include "ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
extern template class ActionGroups<optical::CoreParams, optical::CoreState>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
