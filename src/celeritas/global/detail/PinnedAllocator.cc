//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/PinnedAllocator.cc
//---------------------------------------------------------------------------//

#include "corecel/data/PinnedAllocator.t.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/user/DetectorSteps.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Explicit instantiations
template struct PinnedAllocator<Real3>;
template struct PinnedAllocator<DetectorStepPointOutput::Energy>;
template struct PinnedAllocator<DetectorId>;
template struct PinnedAllocator<ThreadId>;
template struct PinnedAllocator<TrackId>;
template struct PinnedAllocator<EventId>;
template struct PinnedAllocator<ParticleId>;
//---------------------------------------------------------------------------//
}  // namespace celeritas