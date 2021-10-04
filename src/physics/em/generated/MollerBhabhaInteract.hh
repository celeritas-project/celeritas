//----------------------------------*-hh-*-----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "celeritas_config.h"
#include "base/Assert.hh"
#include "../detail/MollerBhabha.hh"

namespace celeritas
{
namespace generated
{
void moller_bhabha_interact(
    const detail::MollerBhabhaHostRef&,
    const ModelInteractRefs<MemSpace::host>&);

void moller_bhabha_interact(
    const detail::MollerBhabhaDeviceRef&,
    const ModelInteractRefs<MemSpace::device>&);

#if !CELERITAS_USE_CUDA
inline void moller_bhabha_interact(
    const detail::MollerBhabhaDeviceRef&,
    const ModelInteractRefs<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace generated
} // namespace celeritas
