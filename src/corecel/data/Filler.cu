//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Filler.cu
//---------------------------------------------------------------------------//
#include "Filler.device.t.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template class Filler<real_type, MemSpace::device>;
template class Filler<size_type, MemSpace::device>;
template class Filler<int, MemSpace::device>;
template class Filler<TrackSlotId, MemSpace::device>;
//---------------------------------------------------------------------------//
}  // namespace celeritas
