//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "ModelBuilder.hh"
#include "PhysicsData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
class PhysicsParams final : public ParamsDataInterface<PhysicsParamsData>
{
  public:
    using SPConstModelBuilder = std::shared_ptr<ModelBuilder const>;

    struct Input
    {
        std::vector<SPConstModelBuilder> model_builders;
    };

  public:
    explicit PhysicsParams(Input)
    {
        HostVal<PhysicsParamsData> data;
        data_ = CollectionMirror<PhysicsParamsData>{std::move(data)};
    }

    //! Access optical physics data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical physics data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    CollectionMirror<PhysicsParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
