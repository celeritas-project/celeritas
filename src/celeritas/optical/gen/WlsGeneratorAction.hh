//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/WlsGeneratorAction.hh
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
namespace optical
{
class CoreParams;
class CoreStateBase;
class WavelengthShiftModel;

//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical wavelength shifting distribution data.
 *
 * This samples and initializes optical photons directly in a track slot in a
 * reproducible way. At each step of the optical loop:
 * - The distribution buffer is compacted to remove invalid distributions
 * - An inclusive scan over \c num_photons is computed. All distributions up to
 *   the buffer size are guaranteed to have a positive photon count.
 * - Each thread is assigned to exactly one photon via the cumulative sum \c
 *   offsets array and generates one WLS secondary in a vacant track slot.
 * - The photon count in the distribution is decremented atomically after
 *   generation. Because the thread to distribution mapping is determined from
 *   the pre-generation cumulative sum, \c num_photons can only reach zero
 *   after all threads assigned to a distribution have generated from it. The
 *   updated counts are used solely to identify exhausted distributions during
 *   the subsequent compaction.
 * - The buffer is compacted again to free space for new distributions that may
 *   be added during the step.
 *
 * This differs from the Cherenkov/scintillation generation in a few ways:
 * - New WLS distributions can be added during the optical stepping loop, so
 *   the buffer must be compacted each step to reclaim space.
 * - Because the buffer is compacted before generation, there are no
 *   zero-photon distributions in the active portion of the buffer; the
 *   executor does not need to skip over them by finding the \c buffer_start
 *   offset.
 * - The number of photons generated in a WLS interaction is small (O(1))
 *   compared to scintillation, which can be large (O(1000)), so there should
 *   be very few collisions in the atomic update and it should have no
 *   performance impact.
 */
class WlsGeneratorAction final : public GeneratorBase
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstWavelengthShiftModel
        = std::shared_ptr<WavelengthShiftModel const>;
    //!@}

  public:
    struct Input
    {
        ActionId action_id;
        AuxId aux_id;
        GeneratorId gen_id;
        SPConstWavelengthShiftModel wls;
        SPConstWavelengthShiftModel wls2;
        size_type capacity{};
    };

    // Construct with IDs, WLS model, and buffer capacity
    explicit WlsGeneratorAction(Input&&);

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

    SPConstWavelengthShiftModel wls_;
    SPConstWavelengthShiftModel wls2_;
    size_type capacity_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
