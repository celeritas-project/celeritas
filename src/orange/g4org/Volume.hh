//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Volume.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/io/Label.hh"
#include "geocel/GeoParamsInterface.hh"
#include "orange/OrangeTypes.hh"
#include "orange/transform/VariantTransform.hh"

class G4LogicalVolume;

namespace celeritas
{
namespace orangeinp
{
class ObjectInterface;
}

namespace g4org
{
//---------------------------------------------------------------------------//
struct LogicalVolume;

//---------------------------------------------------------------------------//
/*!
 * An unconstructed ORANGE CSG Object with a transform.
 *
 * This holds equivalent information to a Geant4 \c G4VPhysicalVolume, but with
 * \em only ORANGE data structures.
 */
struct PhysicalVolume
{
    using ReplicaId = GeantPhysicalInstance::ReplicaId;

    //! Corresponding Geant4 physical volume
    VolumeInstanceId id;
    //! Replica/parameterization (see GeantGeoParams::id_to_geant TODO)
    ReplicaId replica_id;

    VariantTransform transform;
    std::shared_ptr<LogicalVolume const> lv;
};

//---------------------------------------------------------------------------//
/*!
 * A reusable object that can be turned into a UnitProto or a Material.
 *
 * This holds equivalent information to a Geant4 \c G4LogicalVolume, but with
 * \em only ORANGE data structures.
 */
struct LogicalVolume
{
    using SPConstObject = std::shared_ptr<orangeinp::ObjectInterface const>;

    //! Corresponding Geant4 logical volume
    VolumeId id;
    //! Filled material ID
    GeoMatId material_id;

    //! "Unplaced" parent shape
    SPConstObject solid;
    //! Embedded child volumes
    std::vector<PhysicalVolume> children;
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
