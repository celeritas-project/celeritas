//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file HostDetectorStore.cc
//---------------------------------------------------------------------------//
#include "HostDetectorStore.hh"

#include "base/StackAllocatorView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Constructor.
 */
HostDetectorStore::HostDetectorStore(size_type              buffer_capacity,
                                     const UniformGridData& grid)
    : hit_buffer_(buffer_capacity)
    , tally_grid_(grid)
    , tally_deposition_(UniformGrid(grid).size())
{
}

//---------------------------------------------------------------------------//
/*!
 * Get detector data.
 */
DetectorPointers HostDetectorStore::host_pointers()
{
    DetectorPointers result;
    result.hit_buffer       = hit_buffer_.host_pointers();
    result.tally_grid       = tally_grid_;
    result.tally_deposition = make_span(tally_deposition_);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Bin the buffer onto the grid.
 */
void HostDetectorStore::bin_buffer()
{
    UniformGrid grid(tally_grid_);
    auto hits = StackAllocatorView<Hit>(hit_buffer_.host_pointers()).get();
    CELER_ASSERT(hits.size() > 0);

    // Iterate through hits and add them to grid
    for (const auto& hit : hits)
    {
        real_type z_pos = hit.pos[2];
        size_type bin   = 0;
        if (z_pos <= grid.front())
        {
            bin = 0;
        }
        else if (z_pos >= grid.back())
        {
            bin = grid.size() - 1;
        }
        else
        {
            bin = grid.find(z_pos);
        }
        CELER_ASSERT(bin < tally_deposition_.size());

        tally_deposition_[bin] += hit.energy_deposited.value();
    }

    // Clear the hit buffer
    hit_buffer_.clear();
}

//---------------------------------------------------------------------------//
/*!
 * Finalize the tally result.
 */
std::vector<real_type> HostDetectorStore::finalize(real_type norm)
{
    CELER_EXPECT(norm > 0.0);
    for (auto& v : tally_deposition_)
    {
        v *= norm;
    }
    return tally_deposition_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
