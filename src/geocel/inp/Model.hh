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
#include "geocel/Types.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Define a node and reference child edges in the geometry graph.
 *
 * A given volume instance ID can appear only *once* across all volumes.
 * \todo Instead of this, which allows us to "easily" translate between geant4
 * IDs and instance IDs, we should just have a vector of volume instances here.
 *
 * \todo Add region definitions.
 */
struct Volume
{
    //! Name for the edge
    Label label;
    //! Filled material ID
    GeoMatId material;
    //! Child edges
    std::vector<VolumeInstanceId> children;

    //! True if it has a label
    explicit operator bool() const { return !label.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Define an edge in the geometry graph.
 *
 * The \c volume is the node below this edge, the volume being instantiated.
 */
struct VolumeInstance
{
    //! Name for the edge
    Label label;
    //! Logical volume referenced by this instance
    VolumeId volume;

    // TODO: replica numbers

    //! True if it has a label and ID
    explicit operator bool() const { return volume && !label.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Define a graph of geometry elements.
 *
 * \todo Construct from in-memory Geant4
 */
struct Volumes
{
    //! Nodes in the graph (logical volumes)
    std::vector<Volume> volumes;
    //! Properties of edges in the graph (physical volumes)
    std::vector<VolumeInstance> volume_instances;

    //! True if at least one node is defined
    explicit operator bool() const { return !volumes.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Set up geometry/material model.
 *
 * The geometry filename should almost always be a GDML path. As a temporary
 * measure we also support loading from a \c .org.json file if the \c
 * StandaloneInput::physics_import is a ROOT file with serialized physics data.
 *
 * Materials, regions, and surfaces may be loaded from the geometry: this is
 * usually done by \c GeantGeoParams::make_model_input .
 */
struct Model
{
    //! Path to GDML file, Geant4 world, or loaded geometry
    std::variant<std::string, G4VPhysicalVolume const*> geometry;

    // TODO: Materials
    // TODO: Regions
    Volumes volumes;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
