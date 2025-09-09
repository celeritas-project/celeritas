//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
class AuxParamsRegistry;
class CoreParams;

namespace detail
{
template<StepPoint P>
class StepGatherAction;
class StepParams;
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
 *
 * \todo Like the optical collector, this class is not used after it's created:
 * it just serves to create helper classes. Perhaps move to the \c setup
 * namespace?
 */
class StepCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPStepInterface = std::shared_ptr<StepInterface>;
    using SPConstCoreGeo = std::shared_ptr<CoreGeoParams const>;
    using VecInterface = std::vector<SPStepInterface>;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<StepCollector>
    make_and_insert(CoreParams const& core, VecInterface callbacks);

    // Construct with options and register pre/post-step actions
    StepCollector(SPConstCoreGeo geo,
                  VecInterface&& callbacks,
                  AuxParamsRegistry* aux_registry,
                  ActionRegistry* action_registry);

    // See which data are being gathered
    StepSelection const& selection() const;

  private:
    template<StepPoint P>
    using SPStepGatherAction = std::shared_ptr<detail::StepGatherAction<P>>;

    std::shared_ptr<detail::StepParams> params_;
    SPStepGatherAction<StepPoint::pre> pre_action_;
    SPStepGatherAction<StepPoint::post> post_action_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
