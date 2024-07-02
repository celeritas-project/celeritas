//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"

#include "OpticalModelBuilder.hh"

namespace celeritas
{
class OpticalModelMfpBuilder;
//---------------------------------------------------------------------------//
/*!
 * Base class for discrete, volumetric optical models.
 *
 * For optical physics, there is precisely one particle (optical photons)
 * and one energy range (optical wavelengths), so only models and no processes
 * are used in optical physics.
 */
class OpticalModel : public ExplicitOpticalActionInterface,
                     public ConcreteAction
{
  public:
    //! Construct the optical model with action parameters
    OpticalModel(ActionId id, std::string label, std::string description)
        : ConcreteAction(id, label, description)
    {}

    //! Virtual destructor for polymorphic deletion
    virtual ~OpticalModel() = default;

    //! Action order for optical models is always post-step
    ActionOrder order() const override final { return ActionOrder::post; }

    //! Build mean free path grid for the given optical material
    virtual void build_mfp(OpticalModelMfpBuilder&) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
