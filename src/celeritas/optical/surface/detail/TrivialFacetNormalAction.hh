//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/TrivialFacetNormalAction.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    TrivialFacetNormalAction ...;
   \endcode
 */
class TrivialFacetNormalAction : public OpticalStepActionInterface,
                                 public ConcreteAction
{
  public:
    TrivialFacetNormalAction(ActionId);

    inline StepActionOrder order() const final
    {
        return StepActionOrder::post;
    }

    void step(CoreParams const&, CoreStateHost&) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
