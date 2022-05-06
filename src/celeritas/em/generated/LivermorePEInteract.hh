//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/LivermorePEInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/em/data/LivermorePEData.hh"

namespace celeritas
{
namespace generated
{
void livermore_pe_interact(
    const celeritas::LivermorePEHostRef&,
    const CoreRef<MemSpace::host>&);

void livermore_pe_interact(
    const celeritas::LivermorePEDeviceRef&,
    const CoreRef<MemSpace::device>&);

#if !CELER_USE_DEVICE
inline void livermore_pe_interact(
    const celeritas::LivermorePEDeviceRef&,
    const CoreRef<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace generated
} // namespace celeritas
