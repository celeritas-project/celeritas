//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ObserverPtr.device.hh
//! \brief Interoperability with Thrust
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/device_ptr.h>

#include "ObserverPtr.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Create an observer pointer from a Thrust device pointer
template<class T>
ObserverPtr<T, MemSpace::device> make_observer(thrust::device_ptr<T> ptr)
{
    return ObserverPtr<T, MemSpace::device>{ptr.get()};
}

//---------------------------------------------------------------------------//
//! Create a Thrust device pointer from an Observer pointer
template<class T>
thrust::device_ptr<T> device_pointer_cast(ObserverPtr<T, MemSpace::device> ptr)
{
    return thrust::device_ptr<T>{ptr.get()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
