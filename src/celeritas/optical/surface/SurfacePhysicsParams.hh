//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/cont/EnumArray.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "SurfaceModel.hh"
#include "SurfacePhysicsData.hh"

namespace celeritas
{
class ActionRegistry;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfacePhysicsParams ...;
   \endcode
 */
class SurfacePhysicsParams final
    : public ParamsDataInterface<SurfacePhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using SurfaceStepArray = EnumArray<SurfacePhysicsStep, T>;

    using VecModelBuilders = std::vector<SurfaceModel::ModelBuilder>;
    //!@}

    struct SurfaceInput
    {
        std::vector<OptMatId> materials;
        std::vector<SurfaceStepArray<SurfaceModelId>> interface_models;
    };

    struct Input
    {
        ActionRegistry* action_reg = nullptr;

        std::vector<SurfaceInput> surfaces;  //!< indexed by GeometricSurfaceId
        SurfaceStepArray<VecModelBuilders> model_builders;
    };

  public:
    explicit SurfacePhysicsParams(Input) {}

    //! Access surface physics data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access surface physics data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    //! Action ID for initializing boundary interactions
    ActionId init_boundary_action() const { return ActionId{}; }

    //! Action ID for finishing boundary interactions
    ActionId post_boundary_action() const { return ActionId{}; }

    std::vector<std::shared_ptr<SurfaceModel>> models(SurfacePhysicsStep) const
    {
        return {};
    }

  private:
    // Host/device storage
    CollectionMirror<SurfacePhysicsParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
