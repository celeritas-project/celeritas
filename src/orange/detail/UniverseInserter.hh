//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UniverseInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Label.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a universe entry.
 */
class UniverseInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<OrangeParamsData>;
    using VecLabel = std::vector<Label>;
    //!@}

  public:
    // Construct from full parameter data
    UniverseInserter(VecLabel* universe_labels,
                     VecLabel* surface_labels,
                     VecLabel* volume_labels,
                     Data* data);

    // Append the number of local surfaces and volumes
    UniverseId operator()(UniverseType type,
                          Label univ_label,
                          VecLabel surface_labels,
                          VecLabel volume_labels);

  private:
    VecLabel* universe_labels_;
    VecLabel* surface_labels_;
    VecLabel* volume_labels_;

    CollectionBuilder<UniverseType, MemSpace::host, UniverseId> types_;
    CollectionBuilder<size_type, MemSpace::host, UniverseId> indices_;
    CollectionBuilder<size_type> surfaces_;
    CollectionBuilder<size_type> volumes_;

    EnumArray<UniverseType, size_type> num_universe_types_{};
    size_type accum_surface_{0};
    size_type accum_volume_{0};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
