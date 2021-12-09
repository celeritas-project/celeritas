//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoKernel.cc
//---------------------------------------------------------------------------//
#include "LDemoKernel.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Sample mean free path and calculate physics step limits.
 */
void pre_step(const ParamsHostRef& params, const StateHostRef& states)
{
    PreStepLauncher<MemSpace::host> launch(params, states);

#pragma omp parallel for
    for (size_type i = 0; i < states.size(); ++i)
    {
        launch(ThreadId{i});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Propagate and process physical changes to the track along the step and
 * select the process/model for discrete interaction.
 */
void along_and_post_step(const ParamsHostRef& params,
                         const StateHostRef&  states)
{
    AlongAndPostStepLauncher<MemSpace::host> launch(params, states);

#pragma omp parallel for
    for (size_type i = 0; i < states.size(); ++i)
    {
        launch(ThreadId{i});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Postprocessing of secondaries and interaction results.
 */
void process_interactions(const ParamsHostRef& params,
                          const StateHostRef&  states)
{
    ProcessInteractionsLauncher<MemSpace::host> launch(params, states);

#pragma omp parallel for
    for (size_type i = 0; i < states.size(); ++i)
    {
        launch(ThreadId{i});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries.
 */
void cleanup(CELER_MAYBE_UNUSED const ParamsHostRef& params,
             const StateHostRef&                     states)
{
    CleanupLauncher<MemSpace::host> launch(params, states);
    launch(ThreadId{0});
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
