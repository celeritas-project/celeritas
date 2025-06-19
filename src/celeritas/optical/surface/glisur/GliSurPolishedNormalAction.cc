//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurPolishedNormalAction.cc
//---------------------------------------------------------------------------//
#include "GliSurPolishedNormalAction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
GliSurPolishedNormalAction::GliSurPolishedNormalAction(ActionId aid,
                                                       Input const& input)
    : ConcreteAction(
          aid, "sample-normal-glisur", "Sample surface normal by GliSur model")
{
    HostValue data;

    auto build_polish = make_builder(&data.polish);
    auto build_polish_table = make_builder(&data.polish_table);

    for (auto const& polishes : input.model_polishes)
    {
        auto table = build_polish.insert_back(polishes.begin(), polishes.end());
        CELER_ASSERT(table.size() == polishes.size());
        build_polish_table.push_back(table);
    }

    CELER_ENSURE(data.polish_table.size() == input.model_polishes.size());
    CELER_ENSURE(data);

    data_ = CollectionMirror<GliSurPolishedNormalData>(std::move{data});
}

void GliSurPolishedNormalAction::step(CoreParams const& params,
                                      CoreStateHost& state) const
{
    auto execute = make_action_thread_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        GliSurPolishedNormalExecutor{this->host_ref()});
    launch_action(execute);
}

#if !CELER_USE_DEVICE
void GliSurPolishedNormalAction::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
