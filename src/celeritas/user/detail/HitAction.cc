//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/HitAction.cc
//---------------------------------------------------------------------------//
#include "HitAction.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID.
 */
HitAction::HitAction(ActionId            id,
                     ActionOrder         order,
                     SPHitInterface      callback,
                     const HitSelection& selection,
                     SPHitBuffer         buffer)
    : id_(id)
    , order_(order)
    , callback_(std::move(callback))
    , buffer_(std::move(buffer))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(order_ < ActionOrder::size_);
    CELER_EXPECT(callback_);
    CELER_EXPECT(buffer_);

    // Construct shared "params" data
    celeritas::HostVal<HitParamsData> host_data;
    host_data.selection    = selection;
    host_data.is_post_step = (order > ActionOrder::along);
    if (!host_data.is_post_step)
    {
        host_data.selection.post_step = false;
    }
    params_ = CollectionMirror<HitParamsData>(std::move(host_data));
}

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void HitAction::execute(CoreHostRef const& data) const
{
    CELER_EXPECT(data);

    auto launch = make_along_step_launcher(
        data, NoData{}, NoData{}, NoData{}, detail::along_step_neutral);
#pragma omp parallel for
    for (size_type i = 0; i < data.states.size(); ++i)
    {
        launch(ThreadId{i});
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
