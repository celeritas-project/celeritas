//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/Filler.cu
//---------------------------------------------------------------------------//
#include "corecel/data/Filler.device.t.hh"
#include "celeritas/optical/WavelengthShiftData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template class Filler<optical::WlsDistributionData, MemSpace::device>;
//---------------------------------------------------------------------------//
}  // namespace celeritas
