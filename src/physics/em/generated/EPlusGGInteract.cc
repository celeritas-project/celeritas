//----------------------------------*-cc-*-----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGInteract.cc
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "../detail/EPlusGGLauncher.hh"

#include "base/Assert.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace generated
{
void eplusgg_interact(
    const detail::EPlusGGHostRef& eplusgg_data,
    const ModelInteractRef<MemSpace::host>& model)
{
    CELER_EXPECT(eplusgg_data);
    CELER_EXPECT(model);

    detail::EPlusGGLauncher<MemSpace::host> launch(eplusgg_data, model);
    #pragma omp parallel for
    for (size_type i = 0; i < model.states.size(); ++i)
    {
        ThreadId tid{i};
        launch(tid);
    }
}

} // namespace generated
} // namespace celeritas
