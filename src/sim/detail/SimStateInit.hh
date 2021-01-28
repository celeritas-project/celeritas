//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimStateInit.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "../SimInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Initialize the sim state on device
void sim_state_init_device(const SimStatePointers& device_ptrs);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
