//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Transporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Types.hh"
#include "corecel/sys/ActionGroups.hh"
#include "celeritas/Types.hh"
#include "celeritas/user/ActionTimes.hh"
#include "celeritas/user/StepTimes.hh"

#include "CoreState.hh"

namespace celeritas
{
template<class P, template<MemSpace M> class S>
class ActionGroups;

namespace optical
{
class CoreParams;
template<MemSpace M>
class CoreState;
class CoreStateBase;

//---------------------------------------------------------------------------//
/*!
 * Transport all pending optical tracks to completion.
 *
 * \note This class must be constructed \em after all optical actions have been
 * added to the action registry.
 */
class Transporter
{
  public:
    //!@{
    //! \name Type aliases
    using CoreStateHost = CoreState<MemSpace::host>;
    using CoreStateDevice = CoreState<MemSpace::device>;
    using SPConstParams = std::shared_ptr<CoreParams const>;
    using SPActionTimes = std::shared_ptr<ActionTimes>;
    using SPStepTimes = std::shared_ptr<StepTimes>;
    using MapStrDbl = ActionTimes::MapStrDbl;
    using VecDbl = StepTimes::VecDbl;
    //!@}

    struct Input
    {
        SPConstParams params;
        SPActionTimes action_times;  //!< Optional
        SPStepTimes step_times;  //!< Optional
    };

  public:
    // Construct with problem parameters and setup options
    explicit Transporter(Input&&);

    // Transport all pending optical tracks
    void operator()(CoreStateBase&) const;

    //! Access the shared params
    SPConstParams const& params() const { return input_.params; }

    // Get the accumulated action times
    MapStrDbl get_action_times(AuxStateVec const&) const;

    // Get the recorded step times
    VecDbl get_step_times(AuxStateVec const&) const;

  private:
    //// TYPES ////

    using ActionGroupsT = ActionGroups<CoreParams, CoreState>;
    using SPActionGroups = std::shared_ptr<ActionGroupsT>;

    //// DATA ////

    Input input_;
    SPActionGroups actions_;

    //// HELPERS ////

    template<MemSpace M>
    void transport_impl(CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
