//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/GeoParamsInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"  // IWYU pragma: export
#include "corecel/io/Label.hh"  // IWYU pragma: export

#include "BoundingBox.hh"  // IWYU pragma: export
#include "Types.hh"

class G4LogicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interface class for accessing host geometry metadata.
 */
class GeoParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstVolumeId = Span<VolumeId const>;
    //!@}

  public:
    //! Whether safety distance calculations are accurate and precise
    virtual bool supports_safety() const = 0;

    //! Outer bounding box of geometry
    virtual BoundingBox const& bbox() const = 0;

    //// VOLUMES ////

    //! Number of volumes
    virtual VolumeId::size_type num_volumes() const = 0;

    //! Get the label for a placed volume ID
    virtual Label const& id_to_label(VolumeId vol_id) const = 0;

    //! Get the volume ID corresponding to a unique name
    virtual VolumeId find_volume(char const* name) const = 0;

    //! Get the volume ID corresponding to a unique name
    virtual VolumeId find_volume(std::string const& name) const = 0;

    //! Get the volume ID corresponding to a unique label
    virtual VolumeId find_volume(Label const& label) const = 0;

    //! Get the volume ID corresponding to a Geant4 logical volume
    virtual VolumeId find_volume(G4LogicalVolume const* volume) const = 0;

    //! Get zero or more volume IDs corresponding to a name
    virtual SpanConstVolumeId find_volumes(std::string const& name) const = 0;

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~GeoParamsInterface() = default;
};

//---------------------------------------------------------------------------//
/*!
 * Interface class for a host geometry that supports surfaces.
 */
class GeoParamsSurfaceInterface : public GeoParamsInterface
{
  public:
    using GeoParamsInterface::id_to_label;

    //! Get the label for a placed volume ID
    virtual Label const& id_to_label(SurfaceId surf_id) const = 0;

    //! Get the surface ID corresponding to a unique label name
    virtual SurfaceId find_surface(std::string const& name) const = 0;

    //! Number of distinct surfaces
    virtual SurfaceId::size_type num_surfaces() const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
