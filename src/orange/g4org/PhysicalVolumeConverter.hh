//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/PhysicalVolumeConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "Volume.hh"

class G4LogicalVolume;
class G4VPhysicalVolume;

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
/*!
 * Construct a "physical volume" and daughters from a Geant4 object.
 *
 * This recurses through the physical volume. It holds a weak-pointer cache of
 * logical volumes already created.
 */
class PhysicalVolumeConverter
{
  public:
    //!@{
    //! \name Type aliases
    using arg_type = G4VPhysicalVolume const&;
    using result_type = PhysicalVolume;
    //!@}

  public:
    // Construct with options
    explicit PhysicalVolumeConverter(bool verbose);

    // Default destructor
    ~PhysicalVolumeConverter();

    // Convert a physical volume
    result_type operator()(arg_type g4world);

  private:
    struct Data;
    struct Builder;

    // Cached data
    std::unique_ptr<Data> data_;
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
