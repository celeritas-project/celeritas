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
 * Base class for optical models. Optical models are always post-step actions.
 *
 * Because optical photons occur only in one energy band, there is only ever
 * on applicable model for a given process. We therefore uniquely define
 * an OpticalProcess for each OpticalModel via the OpticalProcessImpl
 * template. Subclasses to OpticalModel are expected to provide the following
 * static methods to help identify the OpticalProcess:
 *
 * \code
    static std::string process_label();
    constexpr static ImportOpticalProcessClass process_class();
   \endcode
 *
 * The constructor is expected to take an action id and material parameters
 * const ref.
 */
class OpticalModel : public ExplicitOpticalActionInterface
{
  public:
      // Virtual destructor
    virtual ~OpticalModel() {}

    //! Action order for optical models is always post-step.
    ActionOrder order() const override final { return ActionOrder::post; }

    //! Action ID of the optical model.
    ActionId action_id() const override final { return action_id_; }

  protected:
      //! Construct the optical model with a given action id.
    OpticalModel(ActionId id) : action_id_(id) {}

    ActionId action_id_;
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
