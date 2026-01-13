//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/GeantVolumeInstanceMapper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <vector>

#include "corecel/Config.hh"

#include "geocel/Types.hh"

class G4LogicalVolume;
class G4VPhysicalVolume;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Map between Geant4 PV+copy and Celeritas VolumeInstanceId.
 *
 * This uses a Geant4 world to define a set of \c VolumeInstanceId values. The
 * world will always have \c VolumeInstanceId{0}, and further instance IDs are
 * ordered depth-first.
 *
 * The behavior of this class can be surprising for some Geant4 volume types
 * (replica and parameterised) which have internal thread-local state: their
 * "copy number" reflects transformations applied to them.
 * When querying G4PV*, the PV's state will be incorporated into the resulting
 * VI id. When converting a volume instance to a G4PV pointer, the G4PV will be
 * updated locally so that its transformation and copy number reflect the
 * requested volume instance.
 */
class GeantVolumeInstanceMapper
{
  public:
    //!@{
    //! \name Type aliases
    using G4LV = G4LogicalVolume;
    using G4PV = G4VPhysicalVolume;
    using size_type = VolumeInstanceId::size_type;
    //!@}

  public:
    //! Allow empty construction
    GeantVolumeInstanceMapper() = default;

    // Construct from world volume
    explicit GeantVolumeInstanceMapper(G4PV const& world);

    //! Number of volume instances
    size_type size() const { return g4pv_.size(); }

    // Get the volume instance using the pv and its current state
    VolumeInstanceId geant_to_id(G4PV const&) const;

    // Get the volume instance *without* checking replica state
    VolumeInstanceId geant_to_id(G4PV const&, int copy_no) const;

    // Return and (if replica) update the volume from an instance ID.
    G4PV const& id_to_geant(VolumeInstanceId) const;

    // Get the logical volume from an instance ID (no mutate required)
    G4LV const& logical_volume(VolumeInstanceId vi_id) const;

  private:
    //// DATA ////

    // Map volume (without copy number) to starting VolumeInstanceId
    std::unordered_map<G4PV const*, VolumeInstanceId> base_vi_;
    // Geant4 volumes for each VI ID
    std::vector<G4PV const*> g4pv_;

    //// HELPER FUNCTIONS ////

    void update_replica(G4PV&, VolumeInstanceId) const;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4 && !defined(__DOXYGEN__)
inline VolumeInstanceId
GeantVolumeInstanceMapper::geant_to_id(G4PV const&) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline auto GeantVolumeInstanceMapper::id_to_geant(VolumeInstanceId) const
    -> G4PV const&
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
