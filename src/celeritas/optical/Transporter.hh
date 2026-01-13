//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Transporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Types.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/sys/ActionGroups.hh"
#include "celeritas/Types.hh"
#include "celeritas/user/ActionTimes.hh"

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
    using MapStrDbl = ActionTimes::MapStrDbl;
    //!@}

    struct Input
    {
        SPConstParams params;
        SPActionTimes action_times;  //!< Optional
    };

  public:
    // Construct with problem parameters and setup options
    explicit Transporter(Input&&);

    // Transport all pending optical tracks
    void operator()(CoreStateBase&) const;

    //! Access the shared params
    SPConstParams const& params() const { return data_.params; }

    // Get the accumulated action times
    MapStrDbl get_action_times(AuxStateVec const&) const;

  private:
    //// TYPES ////

    using ActionGroupsT = ActionGroups<CoreParams, CoreState>;
    using SPActionGroups = std::shared_ptr<ActionGroupsT>;

    //// DATA ////

    Input data_;
    SPActionGroups actions_;

    //// HELPERS ////

    template<MemSpace M>
    void transport_impl(CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
