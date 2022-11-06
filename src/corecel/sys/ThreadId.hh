//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ThreadId.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Forward declare meaningless struct to avoid conflict with globally
// namespaced PTL::Thread class when defining ThreadId type.
struct Thread;

//---------------------------------------------------------------------------//
//! Index of a thread inside the current kernel
using ThreadId = OpaqueId<struct Thread>;

//---------------------------------------------------------------------------//
} // namespace celeritas
