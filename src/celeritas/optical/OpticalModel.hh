//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalModel.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
class MaterialParams;

//---------------------------------------------------------------------------//
/*!
 * Base class for optical models. 
 */
class OpticalModel : public ExplicitActionInterface
{
  public:

      virtual ~OpticalModel();

      ActionOrder order() const override final { return ActionOrder::post; }
      ActionId action_id() const override final { return action_id_; }

  protected:
      OpticalModel(ActionId id);

      ActionId action_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
OpticalModel::OpticalModel(ActionId id)
    : action_id_(id)
{}
//---------------------------------------------------------------------------//
}  // namespace celeritas
