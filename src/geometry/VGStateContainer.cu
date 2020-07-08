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
#include "VGStateView.hh"
#include "detail/VGNavStateContainer.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct VGStateContainerPimpl
{
    int                           vgmaxdepth;
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
    state_data_->vgmaxdepth = max_depth;
    state_data_->vgstate = detail::VGNavStateContainer(size, max_depth);
    state_data_->vgnext  = detail::VGNavStateContainer(size, max_depth);
    state_data_->pos.resize(state_size_);
    state_data_->dir.resize(state_size_);
    state_data_->next_step.resize(state_size_);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get a view to on-device states
 */
VGStateView VGStateContainer::device_view() const
{
    REQUIRE(state_data_);

    using thrust::raw_pointer_cast;

    VGStateView result;
    result.size       = state_size_;
    result.vgmaxdepth = state_data_->vgmaxdepth;
    result.vgstate    = state_data_->vgstate.device_view();
    result.vgnext     = state_data_->vgnext.device_view();
    result.pos        = raw_pointer_cast(state_data_->pos.data());
    result.dir        = raw_pointer_cast(state_data_->dir.data());
    result.next_step  = raw_pointer_cast(state_data_->next_step.data());

    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
