//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Model.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

#include "action/ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
class MfpBuilder;

//---------------------------------------------------------------------------//
/*!
 * Base class for discrete, volumetric optical models.
 *
 * For optical physics, there is precisely one particle (optical photons)
 * and one energy range (optical wavelengths), so only models and no processes
 * are used in optical physics.
 */
class Model : public OpticalStepActionInterface, public ConcreteAction
{
  public:
    using ConcreteAction::ConcreteAction;

    //! Action order for optical models is always post-step
    StepActionOrder order() const override final
    {
        return StepActionOrder::post;
    }

    //! Build mean free path grids for all optical materials
    virtual void build_mfps(OpticalMaterialId mat, MfpBuilder& build) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
