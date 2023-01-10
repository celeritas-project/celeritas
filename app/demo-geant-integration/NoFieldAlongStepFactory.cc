//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/NoFieldAlongStepFactory.cc
//---------------------------------------------------------------------------//
#include "NoFieldAlongStepFactory.hh"

#include <memory>
#include <type_traits>

#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/io/ImportData.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Emit the along-step action.
 */
auto NoFieldAlongStepFactory::operator()(argument_type input) const
    -> result_type
{
    // Create along-step action
    return celeritas::AlongStepGeneralLinearAction::from_params(
        input.action_id,
        *input.material,
        *input.particle,
        *input.physics,
        input.imported->em_params.energy_loss_fluct);
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
