//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/Types.hh"
#include "celeritas/geo/GeoFwd.hh"

#include "StepInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;

namespace detail
{
template<StepPoint P>
class StepGatherAction;
struct StepStorage;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Gather and transfer track states at each step.
 *
 * This defines the interface to set up and manage a generic class for
 * interfacing with the GPU track states at the beginning and/or end of every
 * step.
 *
 * \todo The step collector serves two purposes: supporting "sensitive
 * detectors" (mapping volume IDs to detector IDs and ignoring unmapped
 * volumes) and supporting unfiltered output for "MC truth" . Right now only
 * one or the other can be used, not both.
 */
class StepCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPStepInterface = std::shared_ptr<StepInterface>;
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    using VecInterface = std::vector<SPStepInterface>;
    //!@}

  public:
    // Construct with options and register pre/post-step actions
    StepCollector(VecInterface callbacks,
                  SPConstGeo geo,
                  size_type max_streams,
                  ActionRegistry* action_registry);

    // Default destructor and move and copy
    ~StepCollector();
    StepCollector(StepCollector const&);
    StepCollector& operator=(StepCollector const&);
    StepCollector(StepCollector&&);
    StepCollector& operator=(StepCollector&&);

    // See which data are being gathered
    StepSelection const& selection() const;

  private:
    template<StepPoint P>
    using SPStepGatherAction = std::shared_ptr<detail::StepGatherAction<P>>;
    using SPStepStorage = std::shared_ptr<detail::StepStorage>;

    SPStepStorage storage_;
    SPStepGatherAction<StepPoint::pre> pre_action_;
    SPStepGatherAction<StepPoint::post> post_action_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
