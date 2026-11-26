//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/WoodcockParams.hh
//! \sa celeritas/woodcock/Woodcock.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/geo/GeoFwd.hh"

#include "WoodcockData.hh"

namespace celeritas
{
class GeoMaterialParams;
class MaterialParams;
class ParticleParams;
class PhysicsParams;
class VolumeParams;

namespace inp
{
struct Regions;
struct WoodcockTracking;
}  // namespace inp

//---------------------------------------------------------------------------//
/*!
 * Set up Woodcock tracking of photons in user-selected regions.
 *
 * Each selected region is a set of volumes. A volume is a tracking \em root if
 * it belongs to a selected region and is the world volume or is placed inside
 * a volume that does not belong to one. While inside the outermost root in its
 * volume path, a photon is tracked to the root's boundary without stopping at
 * any of the volumes inside. This includes nested volumes that do not belong
 * to a selected region. The majorant cross section bounds the total
 * macroscopic cross section of every material that can be found inside the
 * root.
 *
 * \sa detail::MajorantBuilder
 */
class WoodcockParams final : public ParamsDataInterface<WoodcockParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCoreGeo = std::shared_ptr<CoreGeoParams const>;
    using SPConstGeoMaterial = std::shared_ptr<GeoMaterialParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using SPConstPhysics = std::shared_ptr<PhysicsParams const>;
    using SPConstVolume = std::shared_ptr<VolumeParams const>;
    //!@}

    //! Shared problem data
    struct Input
    {
        SPConstCoreGeo geometry;
        SPConstVolume volume;
        SPConstGeoMaterial geomaterial;
        SPConstMaterial material;
        SPConstParticle particle;
        SPConstPhysics physics;

        //! True if all params are assigned
        explicit operator bool() const
        {
            return geometry && volume && geomaterial && material && particle
                   && physics;
        }
    };

  public:
    // Construct from options and region definitions
    WoodcockParams(Input const& input,
                   inp::WoodcockTracking const& options,
                   inp::Regions const& regions);

    //! Access data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    ParamsDataStore<WoodcockParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
