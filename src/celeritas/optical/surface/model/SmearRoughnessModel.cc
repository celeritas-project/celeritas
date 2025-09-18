//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessModel.cc
//---------------------------------------------------------------------------//
#include "SmearRoughnessModel.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "SmearRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct model from surfaces and inputs.
 */
SmearRoughnessModel::SmearRoughnessModel(SurfaceModelId model,
                                         std::vector<PhysSurfaceId> surfaces,
                                         std::vector<InputT> const& inputs)
    : BuiltinRoughnessModel(model, "smear", std::move(surfaces))
{
    HostVal<SmearRoughnessData> data;
    auto build_roughness = CollectionBuilder{&data.roughness};

    for (auto const& smear : inputs)
    {
        CELER_ENSURE(smear);
        build_roughness.push_back(smear.roughness);
    }

    CELER_ENSURE(data);
    CELER_ENSURE(data.roughness.size() == inputs.size());

    data_ = CollectionMirror<SmearRoughnessData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Launch kernel on host.
 */
void SmearRoughnessModel::step(CoreParams const& params,
                               CoreStateHost& state) const
{
    launch_action(state,
                  this->make_executor(
                      params, state, SmearRoughnessExecutor{data_.host_ref()}));
}

//---------------------------------------------------------------------------//
/*!
 * Launch kernel on device.
 */
#if !CELER_USE_DEVICE
void SmearRoughnessModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
