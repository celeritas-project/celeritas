//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Detector.i.hh
//---------------------------------------------------------------------------//
#include "base/Assert.hh"
#include "base/Types.hh"
#include "physics/grid/UniformGrid.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION Detector::Detector(const Params& params, const State& state)
    : params_(params), state_(state), allocate_(state.hit_buffer)
{
}

//---------------------------------------------------------------------------//
/*!
 * Push the given hit onto the back of the detector stack.
 */
CELER_FUNCTION void Detector::buffer_hit(const Hit& hit)
{
    CELER_EXPECT(hit.thread);
    CELER_EXPECT(hit.time > 0);
    CELER_EXPECT(hit.energy_deposited > celeritas::zero_quantity());

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

    using namespace celeritas;

    const Hit&      hit = state_.hit_buffer.storage[id];
    UniformGrid grid(params_.tally_grid);
    const real_type z_pos = hit.pos[2];
    size_type   bin;

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
} // namespace demo_interactor
