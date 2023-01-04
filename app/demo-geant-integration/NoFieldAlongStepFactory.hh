//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/NoFieldAlongStepFactory.hh
//---------------------------------------------------------------------------//
#pragma once

#include "accel/AlongStepFactory.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct an along-step action with MSC but no field.
 */
class NoFieldAlongStepFactory final
    : public celeritas::AlongStepFactoryInterface
{
  public:
    // Emit the along-step action
    result_type operator()(argument_type input) const final;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
