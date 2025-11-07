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
    using VariantLabel = std::variant<Label, VolumeInstanceId, VolumeId>;
    using VecVarLabel = std::vector<VariantLabel>;
    using VecLabel = std::vector<Label>;
    using SPConstVolumes = std::shared_ptr<VolumeParams const>;
    using VolId = ImplVolumeId;
    using SurfId = ImplSurfaceId;
    //!@}

  public:
    // Construct from full parameter data
    UniverseInserter(SPConstVolumes volume_params,
                     VecLabel* universe_labels,
                     VecLabel* surface_labels,
                     VecLabel* volume_labels,
                     Data* data);

    // Append the number of local surfaces and volumes
    UnivId operator()(UnivType type,
                      Label univ_label,
                      VecLabel surface_labels,
                      VecLabel volume_labels);

    // Append the number of local surfaces and volumes
    UnivId operator()(UnivType type,
                      Label univ_label,
                      VecLabel surface_labels,
                      VecVarLabel volume_labels);

    //!@{
    //! \name Local-to-global mappings for the next universe being built

    //! Next universe
    UnivId next_univ_id() const { return types_.size_id(); }

    //! Get the global ID for the next LocalSurfaceId{0}
    SurfId next_surface_id() const { return SurfId{accum_surface_}; }

    //! Get the global ID for the next LocalVolumeId{0}
    VolId next_volume_id() const { return VolId{accum_volume_}; }

    //!@}

  private:
    // Reference data
    SPConstVolumes volume_params_;

    // Metadata being constructed
    VecLabel* univ_labels_;
    VecLabel* surface_labels_;
    VecLabel* volume_labels_;

    // Data being constructed
    CollectionBuilder<UnivType, MemSpace::host, UnivId> types_;
    CollectionBuilder<size_type, MemSpace::host, UnivId> indices_;
    CollectionBuilder<size_type> surfaces_;
    CollectionBuilder<size_type> volumes_;

    // Optional data being constructed
    CollectionBuilder<VolumeId, MemSpace::host, ImplVolumeId> volume_ids_;
    CollectionBuilder<VolumeInstanceId, MemSpace::host, ImplVolumeId>
        volume_instance_ids_;

    EnumArray<UnivType, size_type> num_univ_types_{};
    SurfId::size_type accum_surface_{0};
    VolId::size_type accum_volume_{0};

    UnivId update_counters(UnivType type,
                           size_type num_surfaces,
                           size_type num_volumes);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
