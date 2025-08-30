//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GaussianRoughnessModel.cc
//---------------------------------------------------------------------------//
#include "GaussianRoughnessModel.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "GaussianRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
GaussianRoughnessModel::GaussianRoughnessModel(
    SurfaceModelId model,
    std::vector<PhysSurfaceId> surfaces,
    std::vector<InputT> const& inputs)
    : BuiltinRoughnessModel(model, "gaussian", std::move(surfaces))
{
    HostVal<GaussianRoughnessData> data;
    auto build_sigma_alpha = ::celeritas::make_builder(&data.sigma_alpha);

    for (auto const& gaussian : inputs)
    {
        CELER_ENSURE(gaussian);
        build_sigma_alpha.push_back(gaussian.sigma_alpha);
    }

    CELER_ENSURE(data);
    CELER_ENSURE(data.sigma_alpha.size() == inputs.size());

    data_ = CollectionMirror<GaussianRoughnessData>{std::move(data)};
}

void GaussianRoughnessModel::step(CoreParams const& params,
                                  CoreStateHost& state) const
{
    launch_action(
        state,
        this->make_executor(
            params, state, GaussianRoughnessExecutorBuilder{data_.host_ref()}));
}

#if !CELER_USE_DEVICE
void GaussianRoughnessModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
