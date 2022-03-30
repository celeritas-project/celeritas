//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file physics/em/generated/KleinNishinaInteract.cc
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "../detail/KleinNishinaLauncher.hh"

#include "base/Assert.hh"
#include "base/Types.hh"
#include "physics/base/InteractionLauncher.hh"

namespace celeritas
{
namespace generated
{
void klein_nishina_interact(
    const celeritas::detail::KleinNishinaHostRef& model_data,
    const CoreRef<MemSpace::host>& core_data)
{
    CELER_EXPECT(core_data);
    CELER_EXPECT(model_data);

    auto launch = make_interaction_launcher(
        core_data.params, core_data.states,
        model_data,
        celeritas::detail::klein_nishina_interact_track);
    #pragma omp parallel for
    for (size_type i = 0; i < core_data.states.size(); ++i)
    {
        ThreadId tid{i};
        launch(tid);
    }
}

} // namespace generated
} // namespace celeritas
