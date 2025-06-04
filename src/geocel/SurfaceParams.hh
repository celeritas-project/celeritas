//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "SurfaceData.hh"

namespace celeritas
{
namespace inp
{
struct Surfaces;
}

//---------------------------------------------------------------------------//
/*!
 * Map volumetric geometry information to surface IDs.
 *
 * This table describes the surface data, its mapping, and the nomenclature
 * in Celeritas and Geant4. Here, "VI" is \c VolumeInstanceId (corresponding to
 * \c G4PhysicalVolume), and "V" is \c VolumeId (corresponding to
 * \c G4LogicalVolume). Interfaces have higher priority than boundaries and
 * therefore lower ID numbers.
 *
 * Data      | Celeritas      | Geant4 | SurfaceId
 * --------- | -------------- | ------ | ---------
 * VI->VI    | Interface      | Border | [0, N_i)
 * V         | Boundary       | Skin   | [N_i, N_i + N_b)
 */
class SurfaceParams final : public ParamsDataInterface<SurfaceParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    // Construct from surface input
    explicit SurfaceParams(inp::Surfaces const&);

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Reference to managed GPU geometry data
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<SurfaceParamsData> data_;
};

//---------------------------------------------------------------------------//

extern template class CollectionMirror<SurfaceParamsData>;
extern template class ParamsDataInterface<SurfaceParamsData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
