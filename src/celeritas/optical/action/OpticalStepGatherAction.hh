//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/OpticalStepGatherAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/ThreadId.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/detail/OpticalStepParams.hh"

#include "ActionInterface.hh"

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
    OpticalStepGatherAction ...;
   \endcode
 */
class OpticalStepGatherAction : public StaticConcreteAction,
                                public OpticalStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPParams = std::shared_ptr<optical::detail::OpticalStepParams>;  //!@}

    // Construct with defaults
    OpticalStepGatherAction(ActionId, SPParams);
    StepActionOrder order() const final { return StepActionOrder::post; }
    void step(CoreParams const&, CoreStateHost&) const final;

#if !CELER_USE_DEVICE
    void step(CoreParams const&, CoreStateDevice&) const final;
#endif

  private:
    SPParams step_params_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
