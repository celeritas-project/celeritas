//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/GeneratorInterface.hh"

#include "GeneratorBase.hh"
#include "GeneratorData.hh"
#include "OffloadData.hh"

namespace celeritas
{
class CoreParams;
class CherenkovParams;
class ScintillationParams;

namespace optical
{
class CoreStateBase;

//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 *
 * This samples and initializes optical photons directly in a track slot in a
 * reproducible way. Multiple threads may generate initializers from a single
 * distribution.
 */
class GeneratorAction final : public GeneratorBase
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstData = Span<GeneratorDistributionData const>;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<GeneratorAction>
    make_and_insert(::celeritas::CoreParams const&,
                    CoreParams const&,
                    size_type capacity);

    // Construct with action ID, data IDs, and optical properties
    GeneratorAction(ActionId, AuxId, GeneratorId, size_type capacity);

    // Add user-provided host distribution data
    void insert(CoreStateBase&, SpanConstData) const;

    //!@{
    //! \name Aux interface

    // Build state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name StepAction interface

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

  private:
    //// DATA ////

    // Starting distribution buffer capacity
    size_type initial_capacity_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void insert_impl(CoreState<M>& state, SpanConstData data) const;

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
