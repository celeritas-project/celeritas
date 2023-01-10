//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"
#include "orange/construct/OrangeInput.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a unit input to params data.
 *
 * Linearize the data in a UnitInput and add it to the host.
 */
class UnitInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<OrangeParamsData>;
    //!@}

  public:
    // Construct from full parameter data
    UnitInserter(Data* orange_data);

    // Create a simple unit and return its ID
    SimpleUnitId operator()(const UnitInput& inp);

  private:
    Data* orange_data_{nullptr};

    // TODO: additional caches for hashed data?

    //// HELPER METHODS ////

    SurfacesRecord insert_surfaces(const SurfaceInput& s);
    VolumeRecord
    insert_volume(const SurfacesRecord& unit, const VolumeInput& v);

    void process_daughter(VolumeRecord*              vol_record,
                          std::vector<Translation>*  translations,
                          const UnitInput::Daughter& daughter);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
