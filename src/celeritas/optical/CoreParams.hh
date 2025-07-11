//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "corecel/Assert.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/random/params/RngParamsFwd.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/user/SDParams.hh"

#include "CoreTrackData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class GeneratorRegistry;
class SurfaceParams;

namespace optical
{
//---------------------------------------------------------------------------//
class MaterialParams;
class PhysicsParams;
//---------------------------------------------------------------------------//
/*!
 * Shared parameters for the optical photon loop.
 */
class CoreParams final : public ParamsDataInterface<CoreParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCoreGeo = std::shared_ptr<CoreGeoParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstPhysics = std::shared_ptr<PhysicsParams const>;
    using SPConstRng = std::shared_ptr<RngParams const>;
    using SPConstSurface = std::shared_ptr<SurfaceParams const>;
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;
    using SPGeneratorRegistry = std::shared_ptr<GeneratorRegistry>;
    using SPConstDetectors = std::shared_ptr<SDParams const>;
    using VecLabel = std::vector<Label>;

    template<MemSpace M>
    using ConstRef = CoreParamsData<Ownership::const_reference, M>;
    template<MemSpace M>
    using ConstPtr = ObserverPtr<ConstRef<M> const, M>;
    //!@}

    struct Input
    {
        SPConstCoreGeo geometry;
        SPConstMaterial material;
        SPConstPhysics physics;
        SPConstRng rng;
        SPConstSurface surface;

        std::optional<VecLabel> detector_labels;

        SPActionRegistry action_reg;
        SPGeneratorRegistry gen_reg;

        //! Maximum number of simultaneous threads/tasks per process
        StreamId::size_type max_streams{1};

        //! True if all params are assigned and valid
        explicit operator bool() const
        {
            return geometry && material && rng && surface && action_reg
                   && gen_reg && max_streams;
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
    SPConstCoreGeo const& geometry() const { return input_.geometry; }
    SPConstMaterial const& material() const { return input_.material; }
    SPConstPhysics const& physics() const { return input_.physics; }
    SPConstRng const& rng() const { return input_.rng; }
    SPConstSurface const& surface() const { return input_.surface; }
    SPActionRegistry const& action_reg() const { return input_.action_reg; }
    SPGeneratorRegistry const& gen_reg() const { return input_.gen_reg; }
    SPConstDetectors const& detectors() const { return detectors_; }
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

    SPConstDetectors detectors_;
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
    else
    {
        CELER_ENSURE(!device_ref_vec_.empty());
        return make_observer(device_ref_vec_);
    }
#if CELER_CUDACC_BUGGY_IF_CONSTEXPR
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
