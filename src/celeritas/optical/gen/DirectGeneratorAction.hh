//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/DirectGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/cont/Span.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/GeneratorInterface.hh"

#include "DirectGeneratorData.hh"
#include "GeneratorBase.hh"

namespace celeritas
{
class CoreParams;

namespace optical
{
class CoreStateBase;

//---------------------------------------------------------------------------//
/*!
 * Generate photons directly from optical track initializers.
 *
 * This generator takes a list of optical track initializers and initializes
 * them directly in a track slot.
 */
class DirectGeneratorAction final : public GeneratorBase
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstData = Span<TrackInitializer const>;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<DirectGeneratorAction>
    make_and_insert(CoreParams const&);

    // Construct with action ID and data IDs
    DirectGeneratorAction(ActionId, AuxId, GeneratorId);

    // Add user-provided host initializer data
    void insert(CoreStateBase&, SpanConstData) const;

    // Build state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void insert_impl(CoreState<M>&, SpanConstData) const;

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
