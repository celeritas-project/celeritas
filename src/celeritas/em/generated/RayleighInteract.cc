//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/RayleighInteract.cc
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "celeritas/em/launcher/RayleighLauncher.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "celeritas/phys/InteractionLauncher.hh"

namespace celeritas
{
namespace generated
{
void rayleigh_interact(
    const celeritas::detail::RayleighHostRef& model_data,
    const CoreRef<MemSpace::host>& core_data)
{
    CELER_EXPECT(core_data);
    CELER_EXPECT(model_data);

    auto launch = make_interaction_launcher(
        core_data,
        model_data,
        celeritas::detail::rayleigh_interact_track);
    #pragma omp parallel for
    for (size_type i = 0; i < core_data.states.size(); ++i)
    {
        ThreadId tid{i};
        launch(tid);
    }
}

} // namespace generated
} // namespace celeritas
