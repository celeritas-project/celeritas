//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GridReflectivityModel.cc
//---------------------------------------------------------------------------//
#include "GridReflectivityModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
GridReflectivityModel::GridReflectivityModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const&)
    : SurfaceModel(id, "reflectivity-grid")
{
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with host data.
 */
void GridReflectivityModel::step(CoreParams const&, CoreStateHost&) const {}

//---------------------------------------------------------------------------//
/*!
 * Execute the model with device data.
 */
#if !CELER_USE_DEVICE
void GridReflectivityModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
