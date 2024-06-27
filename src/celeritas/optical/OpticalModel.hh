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
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    OpticalModel ...;
   \endcode
 */
class OpticalModel : public ExplicitOpticalActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    virtual ~OpticalModel() {}

    //! Action order for optical models is always post-step
    ActionOrder order() const override final { return ActionOrder::post; }

    //! Aciton ID of the optical model.
    ActionId action_id() const override final { return action_id_; }

  protected:
    //! Construct the optical model with a given action id.
    OpticalModel(ActionId id) : action_id_(id) {}

    ActionId action_id_;
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
