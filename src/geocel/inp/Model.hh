//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/inp/Model.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "corecel/io/Label.hh"

#include "Types.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Define a single surface, the boundary around or between volumes.
 *
 * An "interface" surface is an (exiting, entering) pair of volume instances.
 * A "boundary" surface is the entire surface of a volume.
 *
 * See \c SurfaceParams . These are typically loaded from Geant4 via \c
 * celeritas::setup::load_geant .
 */
struct Surface
{
    using Interface = std::pair<VolumeInstanceId, VolumeInstanceId>;
    using Boundary = VolumeId;

    std::variant<Interface, Boundary> surface;
    Label label;
};

//---------------------------------------------------------------------------//
/*!
 * Set up geometry/material model.
 *
 * The geometry filename should almost always be a GDML path. As a temporary
 * measure we also support loading from a \c .org.json file if the \c
 * StandaloneInput::physics_import is a ROOT file with serialized physics data.
 *
 * Materials, regions, and surfaces may be loaded from the geometry.
 */
struct Model
{
    using VecSurface = std::vector<Surface>;

    //! Path to GDML file, or Geant4 world
    std::variant<std::string, G4VPhysicalVolume const*> geometry;

    // TODO: Materials
    // TODO: Regions
    VecSurface surfaces;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
