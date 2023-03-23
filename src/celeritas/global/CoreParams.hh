//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Assert.hh"
#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/random/RngParamsFwd.hh"

#include "ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class AtomicRelaxationParams;
class CutoffParams;
class FluctuationParams;
class GeoMaterialParams;
class MaterialParams;
class ParticleParams;
class PhysicsParams;
class SimParams;
class TrackInitParams;

//---------------------------------------------------------------------------//
/*!
 * Global parameters required to run a problem.
 */
class CoreParams
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstGeoMaterial = std::shared_ptr<GeoMaterialParams const>;
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using SPConstCutoff = std::shared_ptr<CutoffParams const>;
    using SPConstPhysics = std::shared_ptr<PhysicsParams const>;
    using SPConstRng = std::shared_ptr<RngParams const>;
    using SPConstSim = std::shared_ptr<SimParams const>;
    using SPConstTrackInit = std::shared_ptr<TrackInitParams const>;
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;

    using HostRef = HostCRef<CoreParamsData>;
    using DeviceRef = DeviceCRef<CoreParamsData>;
    //!@}

    struct Input
    {
        SPConstGeo geometry;
        SPConstMaterial material;
        SPConstGeoMaterial geomaterial;
        SPConstParticle particle;
        SPConstCutoff cutoff;
        SPConstPhysics physics;
        SPConstRng rng;
        SPConstSim sim;
        SPConstTrackInit init;

        SPActionRegistry action_reg;

        //! Maximum number of simultaneous threads/tasks per process
        StreamId::size_type max_streams{1};

        //! True if all params are assigned
        explicit operator bool() const
        {
            return geometry && material && geomaterial && particle && cutoff
                   && physics && rng && sim && init && action_reg
                   && max_streams;
        }
    };

  public:
    // Construct with all problem data, creating some actions too
    explicit CoreParams(Input inp);

    //!@{
    //! Access shared problem parameter data.
    SPConstGeo const& geometry() const { return input_.geometry; }
    SPConstMaterial const& material() const { return input_.material; }
    SPConstGeoMaterial const& geomaterial() const
    {
        return input_.geomaterial;
    }
    SPConstParticle const& particle() const { return input_.particle; }
    SPConstCutoff const& cutoff() const { return input_.cutoff; }
    SPConstPhysics const& physics() const { return input_.physics; }
    SPConstRng const& rng() const { return input_.rng; }
    SPConstSim const& sim() const { return input_.sim; }
    SPConstTrackInit const& init() const { return input_.init; }
    SPActionRegistry const& action_reg() const { return input_.action_reg; }
    //!@}

    // Access properties on the host
    inline HostRef const& host_ref() const;

    // Access properties on the device
    inline DeviceRef const& device_ref() const;

    //! Maximum number of streams
    size_type max_streams() const { return input_.max_streams; }

  private:
    Input input_;
    HostRef host_ref_;
    DeviceRef device_ref_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access properties on the host.
 */
auto CoreParams::host_ref() const -> HostRef const&
{
    CELER_ENSURE(host_ref_);
    return host_ref_;
}

//---------------------------------------------------------------------------//
/*!
 * Access properties on the device.
 *
 * This will raise an exception if \c celeritas::device is null (and device
 * data wasn't set).
 */
auto CoreParams::device_ref() const -> DeviceRef const&
{
    CELER_ENSURE(device_ref_);
    return device_ref_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
