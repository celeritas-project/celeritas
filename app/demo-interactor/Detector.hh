//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/Detector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/grid/UniformGrid.hh"

#include "DetectorData.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Store a detector hit into the buffer.
 *
 * Input, processing, and resetting must all be done in separate kernels.
 */
class Detector
{
  public:
    //!@{
    //! \name Type aliases
    using Params = DetectorParamsData;
    using State = DetectorStateData<Ownership::reference, MemSpace::native>;
    using SpanConstHits = Span<Hit const>;
    using HitId = OpaqueId<Hit>;
    //!@}

  public:
    // Construct from data references
    inline CELER_FUNCTION Detector(Params const& params, State const& state);

    //// BUFFER INPUT ////

    // Record a hit
    inline CELER_FUNCTION void buffer_hit(Hit const& hit);

    //// BUFFER PROCESSING ////

    // View all hits (*not* for use in same kernel as record/clear)
    inline CELER_FUNCTION HitId::size_type num_hits() const;

    // Process a hit from the buffer to the grid
    inline CELER_FUNCTION void process_hit(HitId id);

    //// BUFFER CLEARING ////

    // Clear the buffer after processing all hits.
    inline CELER_FUNCTION void clear_buffer();

  private:
    Params const& params_;
    State const& state_;
    StackAllocator<Hit> allocate_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION Detector::Detector(Params const& params, State const& state)
    : params_(params), state_(state), allocate_(state.hit_buffer)
{
}

//---------------------------------------------------------------------------//
/*!
 * Push the given hit onto the back of the detector stack.
 */
CELER_FUNCTION void Detector::buffer_hit(Hit const& hit)
{
    CELER_EXPECT(hit.track_slot);
    CELER_EXPECT(hit.time > 0);
    CELER_EXPECT(hit.energy_deposited > zero_quantity());

    // Allocate and assign the given hit
    Hit* allocated = allocate_(1);
    CELER_ASSERT(allocated);
    *allocated = hit;
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of hits in the buffer.
 */
CELER_FUNCTION auto Detector::num_hits() const -> HitId::size_type
{
    return allocate_.get().size();
}

//---------------------------------------------------------------------------//
/*!
 * Bin the given buffered hit into the
 */
CELER_FUNCTION void Detector::process_hit(HitId id)
{
    CELER_EXPECT(id < this->num_hits());

    Hit const& hit = state_.hit_buffer.storage[id];
    UniformGrid grid(params_.tally_grid);
    real_type const z_pos = hit.pos[2];
    size_type bin;

    if (z_pos <= grid.front())
        bin = 0;
    else if (z_pos >= grid.back())
        bin = grid.size() - 1;
    else
        bin = grid.find(z_pos);

    using BinId = ItemId<real_type>;
    atomic_add(&state_.tally_deposition[BinId{bin}],
               hit.energy_deposited.value());
}

//---------------------------------------------------------------------------//
/*!
 * Clear the buffer in a separate kernel after processing.
 */
CELER_FUNCTION void Detector::clear_buffer()
{
    this->allocate_.clear();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
