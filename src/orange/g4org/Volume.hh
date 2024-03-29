//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Volume.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

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
struct LogicalVolume;

//---------------------------------------------------------------------------//
/*!
 * A reusable Object that can be turned into a UnitProto or a Material.
 */
struct LogicalVolume
{
    using SPConstObject = std::shared_ptr<orangeinp::ObjectInterface const>;

    //! Associated Geant4 logical volume
    G4LogicalVolume const* g4lv{nullptr};
    //! Filled material ID
    MaterialId material_id;
    //! "Unplaced" mother shape
    SPConstObject solid;
    //! Logical volume name
    std::string name;
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
