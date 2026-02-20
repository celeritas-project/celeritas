//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalStepInterface.hh
//---------------------------------------------------------------------------//
#pragma once
#include "celeritas/user/StepData.hh"
#include "celeritas/user/StepInterface.hh"

namespace celeritas
{

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Minimal step interface for optical post-step gather.
 *
 * Internal-only: used to activate StepParams allocation
 * for optical step data when enabled via environment variable.
 */
class OpticalStepInterface final : public StepInterface
{
  public:
    //! Return reduced selection (post-step only)
    StepSelection selection() const final
    {
        StepSelection sel;
        sel.points[StepPoint::post].pos = true;
        sel.points[StepPoint::post].volume_id = true;
        return sel;
    }

    //! No detector filters
    Filters filters() const final { return {}; }

    //! No callback processing
    void process_steps(HostStepState) final
    {
        // No-op: optical gather only
    }
    //! Device callback (no-op)
    void process_steps(DeviceStepState) final
    {
        // No-op
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
