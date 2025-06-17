//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/Filler.cu
//---------------------------------------------------------------------------//
#include "corecel/data/Filler.device.t.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template class Filler<TrackStatus, MemSpace::device>;
//---------------------------------------------------------------------------//
}  // namespace celeritas
