//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Model.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "geocel/Types.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Define surfaces, the boundaries between volumes.
 *
 * See \c SurfaceParams . These are typically loaded from Geant4 via \c
 * celeritas::setup::load_geant .
 */
struct Surfaces
{
    using Interface = std::pair<VolumeInstanceId, VolumeInstanceId>;
    using VecInterface = std::vector<VolumeInstanceId>;
    using VecBoundary = std::vector<VolumeId>;

    VecInterface interfaces;
    VecBoundary boundaries;

    //! Whether any surfaces have been specified
    explicit operator bool() const
    {
        return !interfaces.empty() || !boundaries.empty();
    }
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
    //! Path to GDML file, or Geant4 world
    std::variant<std::string, G4VPhysicalVolume const*> geometry;

    // TODO: Materials
    // TODO: Regions
    Surfaces surfaces;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
