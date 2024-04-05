//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Types.hh"
#include "celeritas/optical/OpticalGenData.hh"

#include "detail/PreGenAction.hh"

namespace celeritas
{
class ActionRegistry;
class CerenkovParams;
class OpticalPropertyParams;
class ScintillationParams;

namespace detail
{
struct GenStorage;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Generate scintillation and Cerenkov optical distribution data at each step.
 */
class OpticalCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov = std::shared_ptr<CerenkovParams const>;
    using SPConstProperties = std::shared_ptr<OpticalPropertyParams const>;
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;
    template<MemSpace M>
    using StateRef = OpticalGenStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with optical params, number of streams, and action registry
    OpticalCollector(SPConstProperties properties,
                     SPConstCerenkov cerenkov,
                     SPConstScintillation scintillation,
                     size_type buffer_capacity,
                     size_type num_streams,
                     ActionRegistry* action_registry);

    // Default destructor and move and copy
    ~OpticalCollector() = default;
    CELER_DEFAULT_COPY_MOVE(OpticalCollector);

    // Get stream-local data (throw if not available)
    template<MemSpace M>
    StateRef<M> const& state(StreamId) const;

    //! Get the number of distributions generated for each process
    OpticalBufferOffsets const& num_distributions(StreamId stream) const
    {
        return pregen_action_->num_distributions(stream);
    }

  private:
    //// TYPES ////

    using SPGenStorage = std::shared_ptr<detail::GenStorage>;
    template<StepPoint P>
    using SPPreGenAction = std::shared_ptr<detail::PreGenAction<P>>;

    //// DATA ////

    SPGenStorage storage_;
    SPPreGenAction<StepPoint::pre> gather_action_;
    SPPreGenAction<StepPoint::post> pregen_action_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
