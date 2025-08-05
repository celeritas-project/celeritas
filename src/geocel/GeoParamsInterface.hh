//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoParamsInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Macros.hh"
#include "corecel/cont/LabelIdMultiMap.hh"  // IWYU pragma: export
#include "corecel/cont/Span.hh"  // IWYU pragma: export
#include "corecel/io/Label.hh"  // IWYU pragma: export

#include "BoundingBox.hh"  // IWYU pragma: export
#include "Types.hh"

class G4LogicalVolume;
class G4VPhysicalVolume;

namespace celeritas
{
namespace inp
{
struct Model;
}
//---------------------------------------------------------------------------//
/*!
 * Unique placement/replica of a Geant4 physical volume.
 *
 * This should correspond to a \c VolumeInstanceId and be a unique
 * instantiation of a Geant4 physical volume (PV). Some Geant4 PVs are
 * "parameterised" or "replica" types, which allows a single instance to be
 * mutated at runtime to form a sort of array.
 * If the pointed-to physical volume is *not* a replica/parameterised volume,
 * \c replica is false. Otherwise, it corresponds to the PV's copy number,
 * which can be used to reconstruct the placed volume instance.
 *
 * \todo This will be replaced by a VolumeInstanceId when GeantGeoParams stores
 * a mapping between the canonical geometry definition and its internal
 * parametrised representation.
 */
struct GeantPhysicalInstance
{
    using ReplicaId = OpaqueId<struct Replica_>;

    //! Geant4 physical volume pointer
    G4VPhysicalVolume const* pv{nullptr};
    //! Replica/parameterisation instance
    ReplicaId replica;

    //! False if no PV is associated
    constexpr explicit operator bool() const { return static_cast<bool>(pv); }
};

//---------------------------------------------------------------------------//
/*!
 * Interface class for accessing host geometry metadata.
 *
 * This class is implemented by \c OrangeParams to allow navigation with the
 * ORANGE geometry implementation, \c VecgeomParams for using VecGeom, and \c
 * GeantGeoParams for testing with the Geant4-provided navigator.
 */
class GeoParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstVolumeId = Span<ImplVolumeId const>;
    using ImplVolumeMap = LabelIdMultiMap<ImplVolumeId>;
    using VolInstanceMap = LabelIdMultiMap<VolumeInstanceId>;
    //!@}

  public:
    // Anchor virtual destructor
    virtual ~GeoParamsInterface() = 0;

    //! Whether safety distance calculations are accurate and precise
    virtual bool supports_safety() const = 0;

    //! Outer bounding box of geometry
    virtual BBox const& bbox() const = 0;

    // Create model parameters corresponding to our internal representation
    // TODO: probably will be moved to a separate 'model' class
    virtual inp::Model make_model_input() const = 0;

    //! Get volume metadata
    virtual ImplVolumeMap const& impl_volumes() const = 0;

    //! Get the canonical volume IDs corresponding to an implementation volume
    virtual VolumeId volume_id(ImplVolumeId) const = 0;

    //// TO BE DELETED SOON ////

    //! Get volume instance metadata
    virtual VolInstanceMap const& volume_instances() const = 0;

    //! Get the volume ID corresponding to a Geant4 logical volume
    virtual ImplVolumeId find_volume(G4LogicalVolume const* volume) const = 0;

    //! Get the Geant4 PV corresponding to a volume instance
    virtual GeantPhysicalInstance id_to_geant(VolumeInstanceId id) const = 0;

  protected:
    GeoParamsInterface() = default;
    CELER_DEFAULT_COPY_MOVE(GeoParamsInterface);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
