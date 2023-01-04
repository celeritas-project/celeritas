//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

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
    using SPConstGeo         = std::shared_ptr<const GeoParams>;
    using SPConstMaterial    = std::shared_ptr<const MaterialParams>;
    using SPConstGeoMaterial = std::shared_ptr<const GeoMaterialParams>;
    using SPConstParticle    = std::shared_ptr<const ParticleParams>;
    using SPConstCutoff      = std::shared_ptr<const CutoffParams>;
    using SPConstPhysics     = std::shared_ptr<const PhysicsParams>;
    using SPConstRng         = std::shared_ptr<const RngParams>;
    using SPConstTrackInit   = std::shared_ptr<const TrackInitParams>;
    using SPActionRegistry   = std::shared_ptr<ActionRegistry>;

    using HostRef   = HostCRef<CoreParamsData>;
    using DeviceRef = DeviceCRef<CoreParamsData>;
    //!@}

    struct Input
    {
        SPConstGeo         geometry;
        SPConstMaterial    material;
        SPConstGeoMaterial geomaterial;
        SPConstParticle    particle;
        SPConstCutoff      cutoff;
        SPConstPhysics     physics;
        SPConstRng         rng;
        SPConstTrackInit   init;

        SPActionRegistry action_reg;

        //! True if all params are assigned
        explicit operator bool() const
        {
            return geometry && material && geomaterial && particle && cutoff
                   && physics && rng && init && action_reg;
        }
    };

  public:
    // Construct with all problem data, creating some actions too
    explicit CoreParams(Input inp);

    //!@{
    //! Access shared problem parameter data.
    const SPConstGeo&         geometry() const { return input_.geometry; }
    const SPConstMaterial&    material() const { return input_.material; }
    const SPConstGeoMaterial& geomaterial() const
    {
        return input_.geomaterial;
    }
    const SPConstParticle&  particle() const { return input_.particle; }
    const SPConstCutoff&    cutoff() const { return input_.cutoff; }
    const SPConstPhysics&   physics() const { return input_.physics; }
    const SPConstRng&       rng() const { return input_.rng; }
    const SPConstTrackInit& init() const { return input_.init; }
    const SPActionRegistry& action_reg() const { return input_.action_reg; }
    //!@}

    // Access properties on the host
    inline const HostRef& host_ref() const;

    // Access properties on the device
    inline const DeviceRef& device_ref() const;

  private:
    Input       input_;
    CoreScalars scalars_;
    HostRef     host_ref_;
    DeviceRef   device_ref_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access properties on the host.
 */
auto CoreParams::host_ref() const -> const HostRef&
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
auto CoreParams::device_ref() const -> const DeviceRef&
{
    CELER_ENSURE(device_ref_);
    return device_ref_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
