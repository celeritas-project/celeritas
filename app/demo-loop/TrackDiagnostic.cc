//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackDiagnostic.cc
//---------------------------------------------------------------------------//
#include "TrackDiagnostic.hh"

#include "base/Macros.hh"

using namespace celeritas;

namespace demo_loop
{
template<>
void TrackDiagnostic<MemSpace::device>::end_step(const StateDataRef& states)
{
// Get the number of tracks in flight.
#if CELERITAS_USE_CUDA
    num_alive_per_step_.push_back(demo_loop::reduce_alive(states));
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

template<>
void TrackDiagnostic<MemSpace::host>::end_step(const StateDataRef&)
{
    CELER_ASSERT_UNREACHABLE();
}
} // namespace demo_loop