//----------------------------------*-cc-*-----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaInteract.cc
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "../detail/KleinNishinaLauncher.hh"

#include "base/Assert.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace generated
{
void klein_nishina_interact(
    const detail::KleinNishinaHostRef& klein_nishina_data,
    const ModelInteractRef<MemSpace::host>& model)
{
    CELER_EXPECT(klein_nishina_data);
    CELER_EXPECT(model);

    detail::KleinNishinaLauncher<MemSpace::host> launch(klein_nishina_data, model);
    #pragma omp parallel for
    for (size_type i = 0; i < model.states.size(); ++i)
    {
        ThreadId tid{i};
        launch(tid);
    }
}

} // namespace generated
} // namespace celeritas
