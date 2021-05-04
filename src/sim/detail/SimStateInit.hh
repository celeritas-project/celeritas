//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimStateInit.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"

namespace celeritas
{
template<Ownership W, MemSpace M>
struct SimStateData;

namespace detail
{
//---------------------------------------------------------------------------//
// Initialize the sim state on device
void sim_state_init(
    const SimStateData<Ownership::reference, MemSpace::device>& data);

//---------------------------------------------------------------------------//
// Initialize the sim state on host
void sim_state_init(
    const SimStateData<Ownership::reference, MemSpace::host>& data);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
