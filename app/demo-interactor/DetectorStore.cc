//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorStore.cc
//---------------------------------------------------------------------------//
#include "DetectorStore.hh"

#include "detail/DetectorUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the given capacity for hits.
 */
DetectorStore::DetectorStore(size_type                  buffer_capacity,
                             const UniformGrid::Params& grid)
    : hit_buffer_(buffer_capacity)
    , tally_grid_(grid)
    , tally_deposition_(UniformGrid(grid).size())
{
}

//---------------------------------------------------------------------------//
/*!
 * Get reference to on-device data.
 */
DetectorPointers DetectorStore::device_pointers()
{
    DetectorPointers result;
    result.hit_buffer       = hit_buffer_.device_pointers();
    result.tally_grid       = tally_grid_;
    result.tally_deposition = tally_deposition_.device_pointers();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to bin the buffer into the grid.
 */
void DetectorStore::bin_buffer()
{
    detail::bin_buffer(this->device_pointers());
    hit_buffer_.clear();
}

//---------------------------------------------------------------------------//
/*!
 * Finalize, copy to CPU, and reset, normalizing with the given value.
 */
std::vector<real_type> DetectorStore::finalize(real_type norm)
{
    REQUIRE(norm > 0);
    detail::normalize(this->device_pointers(), norm);
    std::vector<real_type> result(tally_deposition_.size());
    tally_deposition_.copy_to_host(make_span(result));
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
