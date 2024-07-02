//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
class OpticalModel : public ExplicitOpticalActionInterface, public ConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    OpticalModel(ActionId id, std::string label, std::string description)
        : ConcreteAction(id, label, description)
    {}

    //! Virtual destructor for polymorphic deletion
    virtual ~OpticalModel() = default;

    //! Action order for optical models is always post-step
    ActionOrder order() const override final { return ActionOrder::post; }

    //! Get mean free paths for the given optical material
    virtual void build_mfp(OpticalModelMfpBuilder&) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
