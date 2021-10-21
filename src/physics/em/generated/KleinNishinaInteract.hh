//----------------------------------*-hh-*-----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaInteract.hh
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "celeritas_config.h"
#include "base/Assert.hh"
#include "../detail/KleinNishinaData.hh"

namespace celeritas
{
namespace generated
{
void klein_nishina_interact(
    const detail::KleinNishinaHostRef&,
    const ModelInteractRefs<MemSpace::host>&);

void klein_nishina_interact(
    const detail::KleinNishinaDeviceRef&,
    const ModelInteractRefs<MemSpace::device>&);

#if !CELERITAS_USE_CUDA
inline void klein_nishina_interact(
    const detail::KleinNishinaDeviceRef&,
    const ModelInteractRefs<MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace generated
} // namespace celeritas
