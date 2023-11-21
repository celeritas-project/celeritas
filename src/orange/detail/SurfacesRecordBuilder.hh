//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/SurfacesRecordBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "orange/OrangeData.hh"
#include "orange/surf/VariantSurface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a vector of surfaces into type-deleted local surface data.
 *
 * The input surfaces should already be deduplicated.
 */
class SurfacesRecordBuilder
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    using RealId = OpaqueId<real_type>;
    using VecSurface = std::vector<VariantSurface>;
    using result_type = SurfacesRecord;
    //!@}

  public:
    // Construct with pointers to the underlying storage
    SurfacesRecordBuilder(Items<SurfaceType>* types,
                          Items<RealId>* real_ids,
                          Items<real_type>* reals);

    // Construct a record of all the given surfaces
    result_type operator()(VecSurface const& surfaces);

  private:
    CollectionBuilder<SurfaceType> types_;
    CollectionBuilder<RealId> real_ids_;
    DedupeCollectionBuilder<real_type> reals_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
