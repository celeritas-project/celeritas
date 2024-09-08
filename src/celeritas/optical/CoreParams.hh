//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/random/RngParamsFwd.hh"

#include "TrackData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class TrackInitParams;
class SimParams;

namespace optical
{
//---------------------------------------------------------------------------//
class MaterialParams;
// TODO: class PhysicsParams;

//---------------------------------------------------------------------------//
/*!
 * Shared parameters for the optical photon loop.
 */
class CoreParams final : public ParamsDataInterface<CoreParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstRng = std::shared_ptr<RngParams const>;
    using SPConstSim = std::shared_ptr<SimParams const>;
    using SPConstTrackInit = std::shared_ptr<TrackInitParams const>;
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;

    template<MemSpace M>
    using ConstRef = CoreParamsData<Ownership::const_reference, M>;
    template<MemSpace M>
    using ConstPtr = ObserverPtr<ConstRef<M> const, M>;
    //!@}

    struct Input
    {
        SPConstGeo geometry;
        SPConstMaterial material;
        // TODO: physics
        SPConstRng rng;
        SPConstSim sim;
        SPConstTrackInit init;

        SPActionRegistry action_reg;

        //! Maximum number of simultaneous threads/tasks per process
        StreamId::size_type max_streams{1};

        //! True if all params are assigned and valid
        explicit operator bool() const
        {
            return geometry && material && rng && sim && init && action_reg
                   && max_streams;
        }
    };

  public:
    // Construct with all problem data, creating some actions too
    CoreParams(Input&& inp);

    //!@{
    //! \name Data interface
    //! Access data on the host
    HostRef const& host_ref() const final { return host_ref_; }
    //! Access data on the device
    DeviceRef const& device_ref() const final { return device_ref_; }
    //!@}

    //!@{
    //! Access shared problem parameter data.
    SPConstGeo const& geometry() const { return input_.geometry; }
    SPConstMaterial const& material() const { return input_.material; }
    SPConstRng const& rng() const { return input_.rng; }
    SPConstTrackInit const& init() const { return input_.init; }
    SPActionRegistry const& action_reg() const { return input_.action_reg; }
    //!@}

    // Access host pointers to core data
    using ParamsDataInterface<CoreParamsData>::ref;

    // Access a native pointer to properties in the native memory space
    template<MemSpace M>
    inline ConstPtr<M> ptr() const;

    //! Maximum number of streams
    size_type max_streams() const { return input_.max_streams; }

  private:
    Input input_;
    HostRef host_ref_;
    DeviceRef device_ref_;

    // Copy of DeviceRef in device memory
    DeviceVector<DeviceRef> device_ref_vec_;
};

//---------------------------------------------------------------------------//
/*!
 * Access a native pointer to a NativeCRef.
 *
 * This way, CUDA kernels only need to copy a pointer in the kernel arguments,
 * rather than the entire (rather large) DeviceRef object.
 */
template<MemSpace M>
auto CoreParams::ptr() const -> ConstPtr<M>
{
    if constexpr (M == MemSpace::host)
    {
        return make_observer(&host_ref_);
    }
#ifndef __NVCC__
    // CUDA 11.4 complains about 'else if constexpr' ("missing return
    // statement") and GCC 11.2 complains about leaving off the 'else'
    // ("inconsistent deduction for auto return type")
    else
#endif
    {
        CELER_ENSURE(!device_ref_vec_.empty());
        return make_observer(device_ref_vec_);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
