//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/RectArrayInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"

#include "TransformRecordInserter.hh"
#include "../OrangeData.hh"
#include "../OrangeInput.hh"
#include "../OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
class UniverseInserter;
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
    // Construct with universe inserter and parameter data
    RectArrayInserter(UniverseInserter* insert_universe, Data* orange_data);

    // Create a simple unit and return its ID
    UnivId operator()(RectArrayInput const& inp);

  private:
    Data* orange_data_{nullptr};
    TransformRecordInserter insert_transform_;
    UniverseInserter* insert_universe_;

    CollectionBuilder<RectArrayRecord> rect_arrays_;
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<Daughter> daughters_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
