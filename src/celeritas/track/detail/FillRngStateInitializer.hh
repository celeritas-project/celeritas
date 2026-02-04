//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/FillRngStateInitializer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/engine/RanluxppRngEngine.hh"
#include "corecel/random/engine/XorwowRngEngine.hh"

namespace celeritas
{
namespace detail
{

// Fill a XorwowRngEngine state initializer
CELER_FUNCTION
void fillRngStateInitializer(unsigned int seed,
                             unsigned int event_id,
                             unsigned int track_id,
                             unsigned int step_id,
                             XorwowRngEngine::RngStateInitializer_t& rng_init);

// Fill a RanluxppRngEngine state initializer
CELER_FUNCTION
void fillRngStateInitializer(unsigned int seed,
                             unsigned int event_id,
                             unsigned int track_id,
                             unsigned int step_id,
                             RanluxppRngEngine::RngStateInitializer_t& rng_init);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
