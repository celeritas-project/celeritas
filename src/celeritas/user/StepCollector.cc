//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.cc
//---------------------------------------------------------------------------//
#include "StepCollector.hh"

#include <algorithm>
#include <map>
#include <type_traits>
#include <utility>

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/global/CoreParams.hh"

#include "StepInterface.hh"

#include "detail/StepGatherAction.hh"
#include "detail/StepParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<StepCollector>
StepCollector::make_and_insert(CoreParams const& core, VecInterface callbacks)
{
    return std::make_shared<StepCollector>(core.geometry(),
                                           std::move(callbacks),
                                           core.aux_reg().get(),
                                           core.action_reg().get());
}

//---------------------------------------------------------------------------//
/*!
 * Construct with options and register pre and/or post-step actions.
 */
StepCollector::StepCollector(SPConstGeo geo,
                             VecInterface&& callbacks,
                             AuxParamsRegistry* aux_registry,
                             ActionRegistry* action_registry)
{
    CELER_EXPECT(!callbacks.empty());
    CELER_EXPECT(
        std::all_of(callbacks.begin(), callbacks.end(), LogicalTrue{}));
    CELER_EXPECT(geo);
    CELER_EXPECT(aux_registry);
    CELER_EXPECT(action_registry);

    params_ = std::make_shared<detail::StepParams>(
        aux_registry->next_id(), *geo, callbacks);
    aux_registry->insert(params_);

    if (this->selection().points[StepPoint::pre] || params_->has_detectors())
    {
        // Some pre-step data is being gathered
        pre_action_
            = std::make_shared<detail::StepGatherAction<StepPoint::pre>>(
                action_registry->next_id(), params_, VecInterface{});
        action_registry->insert(pre_action_);
    }

    // Always add post-step action, and add callbacks to it
    post_action_ = std::make_shared<detail::StepGatherAction<StepPoint::post>>(
        action_registry->next_id(), params_, std::move(callbacks));
    action_registry->insert(post_action_);
}

//---------------------------------------------------------------------------//
/*!
 * See which data are being gathered.
 */
StepSelection const& StepCollector::selection() const
{
    CELER_EXPECT(params_);
    return params_->selection();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
