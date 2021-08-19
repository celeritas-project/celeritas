//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishina.cc
//---------------------------------------------------------------------------//
#include "KleinNishina.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Interact using the Klein-Nishina model on applicable tracks.
 */
void klein_nishina_interact(const KleinNishinaPointers&              kn,
                            const ModelInteractRefs<MemSpace::host>& model)
{
    KleinNishinaLauncher<MemSpace::host> launch(kn, model);

    for (auto tid : range(ThreadId{model.states.size()}))
    {
        launch(tid);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
