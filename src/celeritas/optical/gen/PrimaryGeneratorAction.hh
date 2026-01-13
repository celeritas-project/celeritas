//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/random/data/DistributionData.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/GeneratorInterface.hh"

#include "GeneratorBase.hh"
#include "OffloadData.hh"
#include "PrimaryGeneratorData.hh"

namespace celeritas
{
class CoreParams;

namespace optical
{
class CoreStateBase;

//---------------------------------------------------------------------------//
/*!
 * Generate optical primaries from user-configurable distributions.
 *
 * This reproducibly samples and initializes optical photons directly in track
 * slots.
 */
class PrimaryGeneratorAction final : public GeneratorBase
{
  public:
    //!@{
    //! \name Type aliases
    using Input = inp::OpticalPrimaryGenerator;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<PrimaryGeneratorAction>
    make_and_insert(CoreParams const&, Input&&);

    // Construct with IDs and distributions
    PrimaryGeneratorAction(ActionId, AuxId, GeneratorId, Input);

    // Set the number of pending tracks
    void insert(CoreStateBase&) const;

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

    PrimaryDistributionData data_;
    CollectionMirror<DistributionParamsData> params_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void insert_impl(CoreState<M>&) const;

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
