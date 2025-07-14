//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UniverseInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Label.hh"

#include "../OrangeData.hh"
#include "../OrangeInput.hh"
#include "../OrangeTypes.hh"

namespace celeritas
{
class VolumeParams;

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
    using VecVolLabel = std::vector<VolumeInput::VariantLabel>;
    using VecVolInst = std::vector<VolumeInstanceId>;
    using VecLabel = std::vector<Label>;
    //!@}

  public:
    // Construct from full parameter data
    UniverseInserter(VolumeParams const& volume_params,
                     VecLabel* universe_labels,
                     VecLabel* surface_labels,
                     VecLabel* volume_labels,
                     Data* data);

    // Append the number of local surfaces and volumes
    UniverseId operator()(UniverseType type,
                          Label univ_label,
                          VecLabel surface_labels,
                          VecLabel volume_labels);

    // Append the number of local surfaces and volumes
    UniverseId operator()(UniverseType type,
                          Label univ_label,
                          VecLabel surface_labels,
                          VecVolLabel volume_labels);

  private:
    // Reference data
    VolumeParams const& volume_params_;

    // Metadata being constructed
    VecLabel* universe_labels_;
    VecLabel* surface_labels_;
    VecLabel* volume_labels_;

    // Data being constructed
    CollectionBuilder<UniverseType, MemSpace::host, UniverseId> types_;
    CollectionBuilder<size_type, MemSpace::host, UniverseId> indices_;
    CollectionBuilder<size_type> surfaces_;
    CollectionBuilder<size_type> volumes_;

    // Optional data being constructed
    CollectionBuilder<VolumeId, MemSpace::host, ImplVolumeId> volume_ids_;
    CollectionBuilder<VolumeInstanceId, MemSpace::host, ImplVolumeId>
        volume_instance_ids_;

    EnumArray<UniverseType, size_type> num_universe_types_{};
    size_type accum_surface_{0};
    size_type accum_volume_{0};

    UniverseId update_counters(UniverseType type,
                               size_type num_surfaces,
                               size_type num_volumes);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
