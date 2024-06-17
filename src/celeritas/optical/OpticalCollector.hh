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

#include "detail/CerenkovPreGenAction.hh"
#include "detail/PreGenGatherAction.hh"
#include "detail/ScintPreGenAction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class CerenkovParams;
class OpticalPropertyParams;
class ScintillationParams;
class CoreParams;

namespace detail
{
struct OpticalGenStorage;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Generate scintillation and Cerenkov optical distribution data at each step.
 *
 * This class is the interface between the main stepping loop and the photon
 * stepping loop and constructs kernel actions for:
 * - gathering the pre-step data needed to generate the optical distributions
 * - generating the optical distributions at the end of the step
 * - launching the photon stepping loop
 *
 * The "collector" (TODO: rename?) will "own" the optical state data and
 * optical params since it's the only thing that launches the optical stepping
 * loop.
 */
class OpticalCollector
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCerenkov = std::shared_ptr<CerenkovParams const>;
    using SPConstCore = std::shared_ptr<CoreParams const>;
    using SPConstProperties = std::shared_ptr<OpticalPropertyParams const>;
    using SPConstScintillation = std::shared_ptr<ScintillationParams const>;
    using SPGenStorage = std::shared_ptr<detail::OpticalGenStorage>;
    //!@}

    struct Input
    {
        //! Optical physics properties for materials
        SPConstProperties properties;
        SPConstCerenkov cerenkov;
        SPConstScintillation scintillation;

        //! Number of steps that have created optical particles
        size_type buffer_capacity{};

        //! True if all input is assigned and valid
        explicit operator bool() const
        {
            return (scintillation || (cerenkov && properties))
                   && buffer_capacity > 0;
        }
    };

  public:
    // Construct with core data and optical params
    OpticalCollector(CoreParams const&, Input&&);

    // Default destructor and move and copy
    ~OpticalCollector() = default;
    CELER_DEFAULT_COPY_MOVE(OpticalCollector);

    //! Get the distribution data
    SPGenStorage const& storage() const { return storage_; };

  private:
    //// TYPES ////

    using SPCerenkovPreGenAction
        = std::shared_ptr<detail::CerenkovPreGenAction>;
    using SPScintPreGenAction = std::shared_ptr<detail::ScintPreGenAction>;
    using SPGatherAction = std::shared_ptr<detail::PreGenGatherAction>;

    //// DATA ////

    SPGenStorage storage_;
    SPGatherAction gather_action_;
    SPCerenkovPreGenAction cerenkov_pregen_action_;
    SPScintPreGenAction scint_pregen_action_;

    // TODO: tracking loop launcher
    // TODO: store optical core params and state?
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
