//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/RectArrayInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"
#include "orange/construct/OrangeInput.hh"

#include "TransformInserter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a RectArrayInput a RectArrayRecord.
 */
class RectArrayInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<OrangeParamsData>;
    //!@}

  public:
    // Construct from full parameter data
    RectArrayInserter(Data* orange_data);

    // Create a simple unit and return its ID
    RectArrayId operator()(RectArrayInput const& inp);

  private:
    Data* orange_data_{nullptr};
    TransformInserter insert_transform_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
