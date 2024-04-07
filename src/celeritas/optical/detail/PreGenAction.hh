//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/optical/OpticalDistributionData.hh"
#include "celeritas/optical/OpticalGenData.hh"

namespace celeritas
{
class CerenkovParams;
class OpticalPropertyParams;
class ScintillationParams;

namespace detail
{
struct GenStorage;
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
class PreGenAction final : public ExplicitCoreActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov = std::shared_ptr<CerenkovParams const>;
    using SPConstProperties = std::shared_ptr<OpticalPropertyParams const>;
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;
    using SPGenStorage = std::shared_ptr<detail::GenStorage>;
    //!@}

    struct IsInvalid
    {
        CELER_FUNCTION bool operator()(OpticalDistributionData data) const
        {
            return !data;
        }
    };

  public:
    // Construct with action ID, optical properties, and storage
    PreGenAction(ActionId id,
                 SPConstProperties properties,
                 SPConstCerenkov cerenkov,
                 SPConstScintillation scintillation,
                 SPGenStorage storage);

    // Launch kernel with host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string label() const final { return "optical-pre-generator-post"; }

    // Name of the action (for user output)
    std::string description() const final;

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::post_post; }

    //! Get the number of distributions generated for each process
    OpticalBufferOffsets const& num_distributions(StreamId stream) const
    {
        CELER_EXPECT(stream);
        return offsets_[stream.get()];
    }

  private:
    //// TYPES ////

    template<MemSpace M>
    using ItemsRef
        = Collection<OpticalDistributionData, Ownership::reference, M>;

    //// DATA ////

    ActionId id_;
    SPConstProperties properties_;
    SPConstCerenkov cerenkov_;
    SPConstScintillation scintillation_;
    SPGenStorage storage_;
    mutable std::vector<OpticalBufferOffsets> offsets_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void execute_impl(CoreParams const&, CoreState<M>&) const;

    void pre_generate(CoreParams const&, CoreStateHost&) const;
    void pre_generate(CoreParams const&, CoreStateDevice&) const;

    size_type remove_if_invalid(ItemsRef<MemSpace::host> const&,
                                size_type,
                                size_type,
                                StreamId) const;
    size_type remove_if_invalid(ItemsRef<MemSpace::device> const&,
                                size_type,
                                size_type,
                                StreamId) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas