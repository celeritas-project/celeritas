//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateContainer.cu
//---------------------------------------------------------------------------//
#include "VGStateContainer.hh"

#include <thrust/device_vector.h>
#include "base/Array.hh"
#include "VGDevice.hh"
#include "detail/VGNavStateContainer.cuda.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct VGStateContainerPimpl
{
    detail::VGNavStateContainer   vgstate;
    detail::VGNavStateContainer   vgnext;
    thrust::device_vector<Real3>  pos;
    thrust::device_vector<Real3>  dir;
    thrust::device_vector<double> next_step;
};

//---------------------------------------------------------------------------//
// Default constructor/destructor/move
VGStateContainer::VGStateContainer()                   = default;
VGStateContainer::~VGStateContainer()                  = default;
VGStateContainer::VGStateContainer(VGStateContainer&&) = default;
VGStateContainer& VGStateContainer::operator=(VGStateContainer&&) = default;

//---------------------------------------------------------------------------//
/*!
 * Construct with device geometry and number of state elements.
 */
VGStateContainer::VGStateContainer(constSPVGDevice device_geom, size_type size)
    : device_geom_(std::move(device_geom_)), state_size_(size)
{
    REQUIRE(device_geom_);
    REQUIRE(state_size_ > 0);

    int max_depth = device_geom_->max_depth();

    state_data_          = std::make_unique<VGStateContainerPimpl>();
    state_data_->vgstate = detail::VGNavStateContainer(size, max_depth);
    state_data_->vgnext  = detail::VGNavStateContainer(size, max_depth);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
