//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateContainer.i.cuh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a view to on-device memory
 */
RngStateView RngStateContainer::device_view()
{
    using thrust::raw_pointer_cast;

    RngStateView::Params params;
    params.size = this->size();
    params.rng  = raw_pointer_cast(rng_.data());
    return RngStateView(params);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
