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
 */
class Transporter
{
  public:
    //!@{
    //! \name Type aliases
    using CoreStateHost = CoreState<MemSpace::host>;
    using CoreStateDevice = CoreState<MemSpace::device>;
    using SPConstParams = std::shared_ptr<CoreParams const>;
    //!@}

    struct Input
    {
        SPConstParams params;
        size_type max_step_iters{numeric_limits<size_type>::max()};
    };

  public:
    // Construct with problem parameters and setup options
    explicit Transporter(Input&&);

    // Transport all pending optical tracks
    void operator()(CoreStateBase&) const;

  private:
    //// TYPES ////

    using ActionGroupsT = ActionGroups<CoreParams, CoreState>;
    using SPActionGroups = std::shared_ptr<ActionGroupsT>;

    //// DATA ////

    SPConstParams params_;
    size_type max_step_iters_{};
    SPActionGroups actions_;

    //// HELPERS ////

    template<MemSpace M>
    void transport_impl(CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
