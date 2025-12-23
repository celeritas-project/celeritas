//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
class G4VPhysicalVolume;

namespace celeritas
{
namespace inp
{
struct Model;
}
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
    //!@}

  public:
    // Anchor virtual destructor in GeoInterface.cc
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

  protected:
    GeoParamsInterface() = default;
    CELER_DEFAULT_COPY_MOVE(GeoParamsInterface);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
