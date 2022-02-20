//----------------------------------*-hh-*-----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "base/Assert.hh"
#include "base/Macros.hh"
#include "../detail/LivermorePEData.hh"

namespace celeritas
{
namespace generated
{
void livermore_pe_interact(
    const detail::LivermorePEHostRef&,
    const ModelInteractRef<MemSpace::host>&);

void livermore_pe_interact(
    const detail::LivermorePEDeviceRef&,
    const ModelInteractRef<MemSpace::device>&);

#if !CELER_USE_DEVICE
inline void livermore_pe_interact(
    const detail::LivermorePEDeviceRef&,
    const ModelInteractRef<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace generated
} // namespace celeritas
