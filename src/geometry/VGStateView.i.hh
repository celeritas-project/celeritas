//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateView.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \brief Construct with invariant parameters
 */
VGStateView::VGStateView(const Params& params) : data_(params)
{
    REQUIRE(data_.size > 0);
    REQUIRE(data_.vgmaxdepth > 0);
    REQUIRE(data_.vgstate != nullptr);
    REQUIRE(data_.vgnext != nullptr);
    REQUIRE(data_.pos != nullptr);
    REQUIRE(data_.dir != nullptr);
    REQUIRE(data_.next_step != nullptr);
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the mutable thread-local state.
 */
CELER_FUNCTION auto VGStateView::operator[](ThreadId id) const -> Ref
{
    REQUIRE(id < this->size());
    Ref result;
    result.vgstate   = this->get_nav_state(data_.vgstate, id.get());
    result.vgnext    = this->get_nav_state(data_.vgnext, id.get());
    result.pos       = data_.pos + id.get();
    result.dir       = data_.dir + id.get();
    result.next_step = data_.next_step + id.get();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Determine the pointer to the navigation state for a particular index.
 *
 * When using the "cuda"-namespace navigation state (i.e., compiling with NVCC)
 * it's necessary to transform the raw data pointer into an index.
 */
CELER_FUNCTION auto
VGStateView::get_nav_state(void* state, size_type idx) const -> NavState*
{
    char* ptr = reinterpret_cast<char*>(state);
#ifdef __NVCC__
    ptr += vecgeom::cuda::NavigationState::SizeOf(data_.vgmaxdepth) * idx;
#else
    REQUIRE(idx == 0);
#endif
    return reinterpret_cast<NavState*>(ptr);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
