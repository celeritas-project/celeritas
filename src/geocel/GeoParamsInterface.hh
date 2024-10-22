//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoParamsInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/LabelIdMultiMap.hh"  // IWYU pragma: export
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
    using SpanConstVolumeId = Span<VolumeId const>;
    using VolumeMap = LabelIdMultiMap<VolumeId>;
    //!@}

  public:
    //! Whether safety distance calculations are accurate and precise
    virtual bool supports_safety() const = 0;

    //! Outer bounding box of geometry
    virtual BBox const& bbox() const = 0;

    //// VOLUMES ////

    //! Get volume metadata
    virtual VolumeMap const& volumes() const = 0;

    //! Get the volume ID corresponding to a Geant4 logical volume
    virtual VolumeId find_volume(G4LogicalVolume const* volume) const = 0;

    //// DEPRECATED: remove in v0.6 ////

    //! Number of volumes
    [[deprecated]] VolumeId::size_type num_volumes() const
    {
        return this->volumes().size();
    }

    //! Get the label for a placed volume ID
    [[deprecated]] Label const& id_to_label(VolumeId vol_id) const
    {
        return this->volumes().at(vol_id);
    }

    //! Get the volume ID corresponding to a unique name
    [[deprecated]] VolumeId find_volume(std::string const& name) const
    {
        return this->volumes().find_unique(name);
    }

    //! Get the volume ID corresponding to a unique label
    [[deprecated]] VolumeId find_volume(Label const& label) const
    {
        return this->volumes().find_exact(label);
    }

    //! Get the volume ID corresponding to a unique name
    [[deprecated]] VolumeId find_volume(char const* name) const
    {
        return this->volumes().find_unique(name);
    }

    //! Get the volume ID corresponding to a unique name
    [[deprecated]] SpanConstVolumeId find_volumes(std::string const& name) const
    {
        return this->volumes().find_all(name);
    }

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~GeoParamsInterface() = default;

    GeoParamsInterface() = default;
    CELER_DEFAULT_COPY_MOVE(GeoParamsInterface);
};

//---------------------------------------------------------------------------//
/*!
 * Interface class for a host geometry that supports surfaces.
 *
 * \todo Remove this interface, use empty surface map instead
 */
class GeoParamsSurfaceInterface : public GeoParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceMap = LabelIdMultiMap<SurfaceId>;
    //!@}

  public:
    using GeoParamsInterface::id_to_label;

    //! Get surface metadata
    virtual SurfaceMap const& surfaces() const = 0;

    //// DEPRECATED: remove in v0.6 ////

    //! Get the label for a placed volume ID
    [[deprecated]] Label const& id_to_label(SurfaceId surf_id) const
    {
        return this->surfaces().at(surf_id);
    }

    //! Get the surface ID corresponding to a unique label name
    [[deprecated]] SurfaceId find_surface(std::string const& name) const
    {
        return this->surfaces().find_unique(name);
    }

    //! Number of distinct surfaces
    [[deprecated]] SurfaceId::size_type num_surfaces() const
    {
        return this->surfaces().size();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
