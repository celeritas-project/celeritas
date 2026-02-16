//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/random/params/RngParamsFwd.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/inp/Scoring.hh"

#include "CoreTrackData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class AuxParamsRegistry;
class CherenkovParams;
class DetectorParams;
class GeneratorRegistry;
class OutputRegistry;
class ScintillationParams;
class SurfaceParams;

namespace optical
{
//---------------------------------------------------------------------------//
class MaterialParams;
class PhysicsParams;
class SimParams;
class SurfacePhysicsParams;
//---------------------------------------------------------------------------//
/*!
 * Shared parameters for the optical photon loop.
 */
class CoreParams final : public ParamsDataInterface<CoreParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;
    using SPOutputRegistry = std::shared_ptr<OutputRegistry>;
    using SPGeneratorRegistry = std::shared_ptr<GeneratorRegistry>;
    using SPAuxRegistry = std::shared_ptr<AuxParamsRegistry>;

    using SPConstCoreGeo = std::shared_ptr<CoreGeoParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstPhysics = std::shared_ptr<PhysicsParams const>;
    using SPConstRng = std::shared_ptr<RngParams const>;
    using SPConstSim = std::shared_ptr<SimParams const>;
    using SPConstSurface = std::shared_ptr<SurfaceParams const>;
    using SPConstSurfacePhysics = std::shared_ptr<SurfacePhysicsParams const>;
    using SPConstDetectors = std::shared_ptr<DetectorParams const>;

    using SPConstCherenkov = std::shared_ptr<CherenkovParams const>;
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;

    template<MemSpace M>
    using ConstRef = CoreParamsData<Ownership::const_reference, M>;
    template<MemSpace M>
    using ConstPtr = ObserverPtr<ConstRef<M> const, M>;
    //!@}

    struct Input
    {
        // Registries
        SPActionRegistry action_reg;
        SPOutputRegistry output_reg;
        SPGeneratorRegistry gen_reg;
        SPAuxRegistry aux_reg;  //!< Optional, empty default

        // Problem definition and state
        SPConstCoreGeo geometry;
        SPConstMaterial material;
        SPConstPhysics physics;
        SPConstRng rng;
        SPConstSim sim;
        SPConstSurface surface;
        SPConstSurfacePhysics surface_physics;
        SPConstDetectors detectors;

        inp::OpticalDetector optical_detector;  //!< Optional
        SPConstCherenkov cherenkov;  //!< Optional
        SPConstScintillation scintillation;  //!< Optional

        //! Maximum number of simultaneous threads/tasks per process
        StreamId::size_type max_streams{1};

        //! Per-process state and buffer capacities
        inp::OpticalStateCapacity capacity;

        //! True if all params are assigned and valid
        explicit operator bool() const
        {
            return geometry && material && rng && sim && surface
                   && surface_physics && action_reg && gen_reg && max_streams
                   && capacity.generators > 0 && capacity.tracks > 0
                   && capacity.primaries > 0;
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
    SPConstSim const& sim() const { return input_.sim; }
    SPConstSurface const& surface() const { return input_.surface; }
    SPConstSurfacePhysics const& surface_physics() const
    {
        return input_.surface_physics;
    }
    SPActionRegistry const& action_reg() const { return input_.action_reg; }
    SPOutputRegistry const& output_reg() const { return input_.output_reg; }
    SPAuxRegistry const& aux_reg() const { return input_.aux_reg; }
    SPGeneratorRegistry const& gen_reg() const { return input_.gen_reg; }
    SPConstDetectors const& detectors() const { return input_.detectors; }
    SPConstCherenkov const& cherenkov() const { return input_.cherenkov; }
    SPConstScintillation const& scintillation() const
    {
        return input_.scintillation;
    }
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
